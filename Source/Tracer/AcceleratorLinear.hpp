#pragma once

#include "AcceleratorLinear.h"



//struct A
//{
//    MaterialToPrimList list;
//
//
//};

//MRAY_HYBRID MRAY_CGPU_INLINE
//void GenerateAccelLeafs(Span<AcceleratorLeaf> gLeafs,
//                        Span<const Vector2ui> primBatchOffsets,
//                        )
//{
//
//}

namespace LinearAccelDetail
{



template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::IntersectionCheck(const Ray& ray,
                                                              const Vector2f& tMinMax,
                                                              Float xi,
                                                              const AcceleratorLeaf& l) const
{
    auto IsInRange = [tMinMax](Float newT) -> bool
    {
        return ((newT >= tMinMax[0]) && (newT < tMinMax[1]));
    };

    using Primitive = typename PG::template Primitive<IdentityTransformContext>;
    using Intersection = IntersectionT<PrimHit>;
    using enum PrimTransformType;

    Ray transformedRay;
    if constexpr(PG::TransformLogic == PER_PRIMITIVE_TRANSFORM)
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
                                                 transformKey, l.primitiveKey);

        transformedRay = transformContext.InvApply(ray);
    }
    else
    {
        // Otherwise, the transform is constant local transform,
        // ray is already transformed to local space
        transformedRay = ray;
    }

    Primitive prim = Primitive(TransformContextIdentity{}, primitiveSoA, l.primitiveKey);
    Optional<Intersection> intersection = prim.Intersects(transformedRay, tMinMax);

    // Intersection decisions
    OptionalHitR<PG> result = std::nullopt;
    if(!intersection) return result;
    if(!IsInRange(intersection->t)) return result;
    if(alphaMap)
    {
        const auto& alphaMapV = alphaMap.value();
        // This has alpha map check it
        Vector2 uv = prim.SurfaceParametrization(intersection.hit);
        Float alpha = alphaMapV(uv).value();
        // Stochastic alpha culling
        if(xi > alpha) return result;
    }

    // It is a hit! Update
    result = HitResult
    {
        .materialKey = l.materialKey,
        .primitiveKey = l.primitiveKey,
        .hit = intersection->hit,
        .t = intersection->t
    };
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
AcceleratorLinear<PG, TG>::AcceleratorLinear(const TransDataSoA& tSoA,
                                             const PrimDataSoA& pSoA,
                                             const DataSoA& dataSoA,
                                             AcceleratorKey aId)
    : cullFace(dataSoA.dCullFace[aId.FetchIndexPortion()])
    , alphaMap(dataSoA.dAlphaMaps[aId.FetchIndexPortion()])
    , leafs(ToConstSpan(dataSoA.dLeafs[aId.FetchIndexPortion()]))
    , transformKey(dataSoA.dInstanceTransforms[aId.FetchIndexPortion()])
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
    OptionalHitR<PG> result = std::nullopt;
    for(const AcceleratorLeaf leaf : leafs)
    {
        result = IntersectionCheck(ray, tMM, rng.NextFloat(), leaf);
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
    OptionalHitR<PG> result = std::nullopt;
    for(const AcceleratorLeaf leaf : leafs)
    {
        result = IntersectionCheck(ray, tMinMax, rng.NextFloat(), leaf);
        if(result) break;
    }
    return result;
}

}

template<PrimitiveGroupC PG>
std::string_view AcceleratorGroupLinear<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = AccelGroupTypeName(BaseAcceleratorLinear::TypeName(),
                                                PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupLinear<PG>::AcceleratorGroupLinear(uint32_t accelGroupId,
                                                   BS::thread_pool& tp, GPUSystem& sys,
                                                   const GenericGroupPrimitiveT& pg,
                                                   const AccelWorkGenMap& globalWorkMap)
    : AcceleratorGroupT<PG>(accelGroupId, tp, sys, pg, globalWorkMap)
    , mem(sys.AllGPUs(), 2_MiB, 32_MiB)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::Construct(AccelGroupConstructParams p,
                                           const GPUQueue& queue)
{
    assert(&this->pg == p.primGroup);
    // Instance Types (determined by transform type)
    this->instanceTypeCount = this->DetermineInstanceTypeCount(p);
    this->typeIds.resize(this->instanceTypeCount);
    std::iota(this->typeIds.begin(), this->typeIds.end(), 0);
    // Total instance count (equavilently total surface count)
    this->instanceCount = this->DetermineInstanceCount(p);
    auto linSurfData = this->LinearizeSurfaceData(p, this->instanceCount, this->pg);
    // Find out the concerete accel count and offsets
    auto leafResult = this->DetermineConcereteAccelCount(std::move(linSurfData.instancePrimBatches),
                                                         std::move(linSurfData.primRanges));
    this->concreteAccelCount = static_cast<uint32_t>(leafResult.concereteAccelIndices.size());

    // Generate offset spans
    using PrimKeySpanList = std::vector<Span<PrimitiveKey>>;
    PrimKeySpanList hInstanceLeafs = this->CreateLeafSubspans(dAllLeafs,
                                                              leafResult.perInstanceLeafRanges);

    // Copy these host vectors to GPU
    // For linear accelerator we only need these at GPU memory
    // additionally we will create instance's globalAABB
    MemAlloc::AllocateMultiData(std::tie(dCullFaceFlags,
                                         dAlphaMaps,
                                         dMaterialKeys,
                                         dPrimitiveRanges,
                                         dTransformKeys,
                                         dLeafs,
                                         dAllLeafs),
                                mem,
                                {this->instanceCount, this->instanceCount,
                                 this->instanceCount, this->instanceCount,
                                 this->instanceCount, this->instanceCount,
                                 leafResult.totalLeafCount});

    // Actual memcpy
    Span<CullFaceFlagArray>     hSpanCullFaceFlags(linSurfData.cullFaceFlags);
    Span<AlphaMapArray>         hSpanAlphaMaps(linSurfData.alphaMaps);
    Span<MaterialKeyArray>      hSpanMaterialKeys(linSurfData.materialKeys);
    Span<PrimRangeArray>        hSpanPrimitiveRanges(linSurfData.primRanges);
    Span<TransformKey>          hSpanTransformKeys(linSurfData.transformKeys);
    Span<Span<PrimitiveKey>>    hSpanLeafs(hInstanceLeafs);
    queue.MemcpyAsync(dCullFaceFlags,   ToConstSpan(hSpanCullFaceFlags));
    queue.MemcpyAsync(dAlphaMaps,       ToConstSpan(hSpanAlphaMaps));
    queue.MemcpyAsync(dMaterialKeys,    ToConstSpan(hSpanMaterialKeys));
    queue.MemcpyAsync(dPrimitiveRanges, ToConstSpan(hSpanPrimitiveRanges));
    queue.MemcpyAsync(dTransformKeys,   ToConstSpan(hSpanTransformKeys));
    queue.MemcpyAsync(dLeafs,           ToConstSpan(hSpanLeafs));

    //Instantiate X

}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> aabbWriteRegion,
                                                           Span<AcceleratorKey> keyWriteRegion) const
{


}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::CastLocalRays(// Output
                                               Span<HitKeyPack> dHitIds,
                                               Span<MetaHit> dHitParams,
                                               Span<SurfaceWorkKey> dWorkKeys,
                                               // I-O
                                               Span<BackupRNGState> rngStates,
                                               // Input
                                               Span<const RayGMem> dRays,
                                               Span<const RayIndex> dRayIndices,
                                               Span<const CommonKey> dAccelKeys,
                                               // Constants
                                               uint32_t instanceId,
                                               const GPUQueue& queue)
{

}