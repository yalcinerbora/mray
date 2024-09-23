#include "OptiXPTX.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/PrimitivesDefault.h"

#include "Core/BitFunctions.h"

// ExternCWrapper Macro
#define WRAP_FUCTION_RAYGEN(NAME, FUNCTION) \
    extern "C" __global__ void __raygen__##NAME(){FUNCTION();}

#define WRAP_FUCTION_MISS(NAME, FUNCTION) \
    extern "C" __global__ void __miss__##NAME(){FUNCTION();}

#define WRAP_FUCTION_CLOSEST_HIT(NAME, PG) \
    extern "C" __global__ void __closesthit__##NAME(){KCClosestHit<PG>();}

#define WRAP_FUCTION_ANY_HIT(NAME, PG) \
    extern "C" __global__ void __anyhit__##NAME(){KCAnyHit<PG>();}

#define WRAP_FUCTION_INTERSECT(NAME, PG, TG) \
    extern "C" __global__ void __intersection__##NAME(){KCIntersect<PG,TG>();}

extern "C" __constant__ ArgumentPackOpitX params;

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T* DriverPtrToType(CUdeviceptr ptrInt)
{
    // TODO: std::bit_cast fails when optixir used in
    // module creation (It crashes). Report to OptiX.
    void* hrRaw;
    std::memcpy(&hrRaw, &ptrInt, sizeof(CUdeviceptr));
    return static_cast<T*>(hrRaw);
}

template <VectorC Hit, bool IsTriangle>
requires std::is_floating_point_v<typename Hit::InnerType>
MRAY_GPU MRAY_GPU_INLINE
MetaHit ReadHitFromAttributes()
{
    static_assert(Hit::Dims >= MetaHit::MaxDim,
                  "Could not fit hit into meta hit!");
    Hit h = Hit::Zero();
    if constexpr(4 <= Hit::Dims)
        h[3] = __uint_as_float(optixGetAttribute_3());
    if constexpr(3 <= Hit::Dims)
        h[2] = __uint_as_float(optixGetAttribute_2());
    if constexpr(2 <= Hit::Dims)
        h[1] = __uint_as_float(optixGetAttribute_1());
    if constexpr(1 <= Hit::Dims)
        h[0] = __uint_as_float(optixGetAttribute_0());

    // MRay barycentric order is different
    if constexpr(IsTriangle)
    {
        Float c = Float(1) - h[0] - h[1];
        h = Hit(c, h[0]);
    }
    return MetaHit(h);
}

template <VectorC Hit>
requires std::is_floating_point_v<typename Hit::InnerType>
MRAY_GPU MRAY_GPU_INLINE
void ReportIntersection(const IntersectionT<Hit>& intersection, unsigned int hitKind)
{
    Float t = intersection.t;
    const Hit& h = intersection.hit;

    if constexpr(1 == Hit::Dims)
        optixReportIntersection(t, hitKind,
                                __float_as_uint(float(h[0])));
    else if constexpr(2 == Hit::Dims)
        optixReportIntersection(t, hitKind,
                                __float_as_uint(float(h[0])),
                                __float_as_uint(float(h[1])));
    else if constexpr(3 == Hit::Dims)
        optixReportIntersection(t, hitKind,
                                __float_as_uint(float(h[0])),
                                __float_as_uint(float(h[1])),
                                __float_as_uint(float(h[2])));
    else if constexpr(4 == Hit::Dims)
        optixReportIntersection(t, hitKind,
                                __float_as_uint(float(h[0])),
                                __float_as_uint(float(h[1])),
                                __float_as_uint(float(h[2])),
                                __float_as_uint(float(h[3])));
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
    // it has 32-bit state. So directly writing it as payload.
    // What about generic multi states?
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
//template<PrimitiveGroupC PGroup>
template<class PGroup>
MRAY_GPU MRAY_GPU_INLINE
void KCClosestHit()
{
    using Hit = typename PGroup::Hit;
    using HitRecord = GenericHitRecordData<>;
    const auto& record = *DriverPtrToType<const HitRecord>(optixGetSbtDataPointer());

    const uint32_t leafId = optixGetPrimitiveIndex();
    const uint32_t rayId = optixGetLaunchIndex().x;

    // Fetch the workKey, transformId, primitiveId from table
    PrimitiveKey pKey = record.dPrimKeys[leafId];
    TransformKey tKey = record.transformKey;
    LightOrMatKey lmKey = record.lightOrMatKey;
    AcceleratorKey aKey = record.acceleratorKey;
    MetaHit hit = ReadHitFromAttributes<Hit, TrianglePrimGroupC<PGroup>>();

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
    using HitRecord = GenericHitRecordData<typename PGroup::DataSoA>;
    const auto& record = *DriverPtrToType<const HitRecord>(optixGetSbtDataPointer());

    if(record.alphaMap)
    {
        // This has alpha map check it
        const auto& alphaMap = record.alphaMap.value();
        const uint32_t leafId = optixGetPrimitiveIndex();
        PrimitiveKey pKey = record.dPrimKeys[leafId];
        // Get the current hit
        MetaHit metaHit = ReadHitFromAttributes<Hit, TrianglePrimGroupC<PGroup>>();
        Hit hit = metaHit.AsVector<Hit::Dims>();
        // Create primitive
        Primitive prim(TransformContextIdentity{}, *record.primSoA, pKey);
        // Finally get uv form hit and get alpha
        Vector2 uv = prim.SurfaceParametrization(hit);
        Float alpha = alphaMap(uv).value();
        // Stochastic alpha culling
        BackupRNGState s = GetRNGStateFromPayload();
        Float xi = BackupRNG(s).NextFloat();
        if(xi >= alpha)
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
template<PrimitiveGroupC PGroup, TransformGroupC TGroup,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(PGroup, TGroup)>
MRAY_GPU MRAY_GPU_INLINE
void KCIntersect()
{
    using enum PrimTransformType;
    static constexpr bool IsPerPrimTransform = (PGroup::TransformLogic == PER_PRIMITIVE_TRANSFORM);

    using Primitive = typename PGroup:: template Primitive<>;
    using Intersection = typename Primitive::Intersection;
    using Hit = typename PGroup::Hit;
    using HitRecord = GenericHitRecordData<typename PGroup::DataSoA,
                                           typename TGroup::DataSoA>;
    const auto& record = *DriverPtrToType<const HitRecord>(optixGetSbtDataPointer());

    // Prim Key
    const uint32_t leafId = optixGetPrimitiveIndex();
    PrimitiveKey pKey = record.dPrimKeys[leafId];

    // Get the ray
    float3 rD = optixGetObjectRayDirection();
    float3 rP = optixGetObjectRayOrigin();
    Ray ray = Ray(Vector3(rD.x, rD.y, rD.z), Vector3(rP.x, rP.y, rP.z));
    if constexpr(IsPerPrimTransform)
    {
        // If we are per-prim transform, the object-space transform
        // is the common transform. We need to do the last process here
        using TransContext = typename PrimTransformContextType<PGroup, TGroup>::Result;
        TransContext tContext = GenerateTransformContext(*record.transSoA,
                                                         *record.primSoA,
                                                         record.transKey,
                                                         pKey);
        ray = tContext.InvApply(ray);
    }
    // For constant-local transform, to object is enough
    //
    // Get the actual primitive,
    // We already transformed the ray, so generate via
    // identity transform context
    Primitive prim(TransformContextIdentity{}, *record.primSoA, pKey);
    // TODO: We need back-face culling to transfer here
    Intersection result = prim.Intersects(ray, true);

    if(result) ReportIntersection(*result, 0);
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

    // Set the RNG state as payload, any hit shaders will
    // do stochastic any hit invocation.
    BackupRNGState rngState = params.dRNGStates[rIndex];
    // Set the ray index (indirection) as payload as well
    // so we do not hit GMem for this.
    // Trace!
    optixTrace(// Accelerator
               params.baseAccelerator,
               // Ray Input
               make_float3(ray.Pos()[0], ray.Pos()[1], ray.Pos()[2]),
               make_float3(ray.Dir()[0], ray.Dir()[1], ray.Dir()[2]),
               tMM[0], tMM[1],
               0.0f,
               //
               OptixVisibilityMask(0xFF),
               // Flags
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               // SBT
               0, 1, 0,
               rIndex, rngState);

    // Save the state back
    params.dRNGStates[rIndex] = rngState;
}

// Actual Definitions
WRAP_FUCTION_RAYGEN(OptiX, KCRayGenOptix);
WRAP_FUCTION_MISS(OptiX, KCMissOptiX);
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
WRAP_FUCTION_CLOSEST_HIT(Triangle, PrimGroupTriangle)
WRAP_FUCTION_ANY_HIT(Triangle, PrimGroupTriangle)
//
WRAP_FUCTION_CLOSEST_HIT(TriangleSkinned, PrimGroupSkinnedTriangle)
WRAP_FUCTION_ANY_HIT(TriangleSkinned, PrimGroupSkinnedTriangle)
// Sphere
WRAP_FUCTION_CLOSEST_HIT(Sphere, PrimGroupSphere)
WRAP_FUCTION_ANY_HIT(Sphere, PrimGroupSphere)
WRAP_FUCTION_INTERSECT(Sphere_Identity, PrimGroupSphere, TransformGroupIdentity);
WRAP_FUCTION_INTERSECT(Sphere_Single, PrimGroupSphere, TransformGroupSingle);
// TODO: Add more...
