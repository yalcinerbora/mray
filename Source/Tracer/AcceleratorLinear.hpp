#pragma once

#include "AcceleratorLinear.h"

namespace LinearAccelDetail
{

template<PrimitiveGroupC PG>
using OptionalHitR = Optional<HitResultT<typename PG::Hit>>;

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::IntersectionCheck(const Ray& ray,
                                                              const Vector2f& tMinMax,
                                                              Float xi,
                                                              const AcceleratorLeaf& l) const
{
    auto IsInRange = [tMinMax](Float newT) -> bool
    {
        return ((newT >= rayData.tMin) &&
                (newT < rayData.tMax));
    };

    using Primitive = typename PrimGroup::Primitive<IdentityTransformContext>;
    using Intersection = IntersectionT<PrimHit>;
    using enum PrimTransformType;

    Ray transformedRay;
    if constexpr(PG::TransformType == PER_PRIMITIVE_TRANSFORM)
    {
        // Primitive has per primitive transform
        // (skinned mesh maybe) we need to transform using primitive's data
        // as well.

        // Compile-time find the transform generator function and return type
        static constexpr
        auto TContextGen = AcquireTransformContextGenerator<PG, TG>();
        constexpr auto TGenFunc = decltype(TContextGen)::Function;
        // Define the types
        // First, this kernel uses a transform context
        // that this primitive group provides to generate the prim
        using TContextType = typename decltype(TContextGen)::ReturnType;
        // Define the primitive type

        // The actual context
        TContextType transformContext = TGenFunc(transformSoA, primitiveSoA,
                                                 transformId, l.primitiveId);

        transformedRay = transformContext.InvApply(ray);
    }
    else
    {
        // Otherwise, the transform is constant local transform,
        // ray is already transformed to local space
        transformedRay = ray;
    }

    Primitive prim = Primitive(IdentityTransformContext{}, primData, leaf.primitiveId);
    Optional<Intersection> intersection = p.Intersects(transformedRay, tMinMax);

    // Intersection decisions
    OptionalHitR<PG> result = std::nullopt;
    if(!intersection) return result;
    if(!IsInRange(intersection->t)) return result;
    if(alphaMap)
    {
        // This has alpha map check it
        Vector2 uv = p.SurfaceParametrization(intersection.hit);
        Float alpha = alphaMap(uv);
        // Stochastic alpha culling
        if(xi > alpha) return result;
    }

    // It is a hit! Update
    result = HitResult
    {
        .materialId = leaf.materialId,
        .primitiveId = leaf.primitiveId,
        .hit = intersection->hit,
        .t = intersection->t
    };
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
AcceleratorLinear<PG, TG>::AcceleratorLinear(const TransDataSoA& tSoA,
                                             const PrimDataSoA& pSoA,
                                             const DataSoA& dataSoA,
                                             TransformId tId,
                                             AcceleratorId aId)
    : cullFace(dataSoA.cullFace[aId.FetchIndexPortion()])
    , alphaMap(dataSoA.dAlphaMaps[aId.FetchIndexPortion()])
    , leafs(ToConstSpan(dataSoa.dLeafs[aId.FetchIndexPortion()]))
    , transformId(tId)
    , transformSoA(tSoA)
    , primitiveSoA(pSoA)
{}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::ClosestHit(BackupRNG& rng,
                                                       const Ray& ray,
                                                       const Vector2& tMinMax) const
{
    Vector2 tMM = tMinMax;
    // Linear search over the array
    Optional<HitResult<PT>> result = std::nullopt;
    for(const Leaf leaf : leafs)
    {
        result = IntersectRoutine(ray, tMM, rng.NextFloat(), leaf);
        if(result) tMM[1] = min(tMM[1], result->t);
    }
    return result;
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::FirstHit(BackupRNG& rng,
                                                     const Ray& ray,
                                                     const Vector2f& tMinMax) const
{
    // Linear search over the array
    Optional<HitResult<PT>> result = std::nullopt;
    for(const Leaf leaf : leafs)
    {
        result = IntersectRoutine(ray, tMin, rng.NextFloat(), leaf);
        if(result) break;
    }
    return result;
}

}