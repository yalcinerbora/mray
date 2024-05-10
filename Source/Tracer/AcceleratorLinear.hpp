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
        return ((newT >= tMinMax[0]) && (newT < tMinMax[1]));
    };

    using Primitive = typename PG::template Primitive<IdentityTransformContext>;
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
                                             AcceleratorKey aId,
                                             TransformKey tId)
    : cullFace(dataSoA.dCullFace[aId.FetchIndexPortion()])
    , alphaMap(dataSoA.dAlphaMaps[aId.FetchIndexPortion()])
    , leafs(ToConstSpan(dataSoA.dLeafs[aId.FetchIndexPortion()]))
    , transformKey(tId)
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
    using namespace std::string_view_literals;
    static const auto Name = CreateAcceleratorType(BaseAcceleratorLinear::TypeName(),
                                                   PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupLinear<PG>::AcceleratorGroupLinear(uint32_t accelGroupId,
                                                   const GenericGroupPrimitiveT& pg)
    : accelGroupId(accelGroupId)
    , pg(static_cast<const GenericGroupPrimitiveT&>(pg))
    , concreteAccelCounter(0)
    , surfaceIdCounter(0)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::Construct(AccelGroupConstructParams p)
{
    assert(&pg == p.primGroup);

    size_t surfCount = std::transform_reduce(p.tGroupSurfs.cbegin(),
                                             p.tGroupSurfs.cend(),
                                             size_t(0), std::plus{},
                                             [](const auto& groupedSurf)
    {
        return groupedSurf.second.size();
    });
    size_t lSurfCount = std::transform_reduce(p.tGroupLightSurfs.cbegin(),
                                              p.tGroupLightSurfs.cend(),
                                              size_t(0), std::plus{},
                                              [](const auto& groupedSurf)
    {
        return groupedSurf.second.size();
    });
    size_t totalSurfaceCount = (surfCount + lSurfCount);

    // Do the loading logic on the CPU
    std::vector<PrimRangeArray> hPrimRanges(totalSurfaceCount);
    std::vector<MaterialKeyArray> hMaterialKeys(totalSurfaceCount);
    std::vector<AlphaMapArray> hAlphaMaps(totalSurfaceCount);
    std::vector<CullFaceFlagArray> hCullFaceFlags(totalSurfaceCount);
    std::vector<TransformKey> hTransformKeys(totalSurfaceCount);

    const auto InitRest = [&](size_t restStart, size_t index)
    {
        for(size_t i = restStart;
            i < TracerConstants::MaxPrimBatchPerSurface; i++)
        {
            hAlphaMaps[index][i] = std::nullopt;
            hCullFaceFlags[index][i] = false;
            hMaterialKeys[index][i] == MaterialKey::InvalidKey();
            hPrimRanges[index][i] = Vector2ui(std::numeric_limits<uint32_t>::max());
        }
    };

    uint32_t index = 0;
    for(const auto& surfList : p.tGroupSurfs)
    for(const auto& [_, surf] : surfList.second)
    {
        assert(surf.alphaMaps.size() == surf.cullFaceFlags.size());
        assert(surf.cullFaceFlags.size() == surf.materials.size());
        assert(surf.materials.size() == surf.primBatches.size());
        for(size_t i = 0; i < surf.alphaMaps.size(); i++)
        {
            if(surf.alphaMaps[i].has_value())
            {
                const GenericTextureView& view = p.textureViews->at(surf.alphaMaps[i].value());
                assert(std::holds_alternative<AlphaMap>(view));
                hAlphaMaps[index][i] = std::get<AlphaMap>(view);
            }
            else hAlphaMaps[index][i] = std::nullopt;

            hCullFaceFlags[index][i] = surf.cullFaceFlags[i];
            hMaterialKeys[index][i] == MaterialKey(static_cast<CommonKey>(surf.materials[i]));
            hPrimRanges[index][i] = pg.BatchRange(surf.primBatches[i]);
        }
        InitRest(index, surf.alphaMaps.size());
        hTransformKeys[index] = TransformKey(static_cast<uint32_t>(surf.transformId));
        index++;
    }
    //for(const auto& lSurfList : p.tGroupLightSurfs)
    //for(const auto& [_, surf] : lSurfList.second)
    //{
    //    //surf.
    //    //const auto& view = p.textureViews->at(surf.alphaMaps[i]);
    //    //    assert(std::holds_alternative<TextureView<2, Float>>(view));
    //    //    hAlphaMaps[index][i] = std::get<TextureView<2, Float>>(view);
    //    //    hCullFaceFlags[index][i] = surf.cullFaceFlags[i];
    //    //    hMaterialKeys[index][i] == MaterialKey(static_cast<CommonKey>(surf.materials[i]));
    //    //    hPrimRanges[index][i] = pg.BatchRange(surf.primBatches[i]);
    //    InitRest(index, 0);
    //    hTransformKeys[index] = TransformKey(static_cast<uint32_t>(surf.transformId));
    //    index++;
    //}
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupLinear<PG>::InstanceCount() const
{

}

template<PrimitiveGroupC PG>
uint32_t AcceleratorGroupLinear<PG>::InstanceTypeCount() const
{

}

template<PrimitiveGroupC PG>
uint32_t AcceleratorGroupLinear<PG>::UsedIdBitsInKey() const
{

}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> aabbWriteRegion,
                                                           Span<AcceleratorKey> keyWriteRegion) const
{

}

template<PrimitiveGroupC PG>
uint32_t AcceleratorGroupLinear<PG>::SetKeyOffset(uint32_t offset)
{

}

//
//AcceleratorKey ReserveSurface(const SurfPrimIdList& primIds,
//                              const SurfMatIdList& matIds) override
//{
//    using enum PrimTransformType;

//    // Generate a new accelerator per instance(surface)
//    // only if the transform requirement is locally constant
//    uint32_t concAccelIndex = concreteAccelCounter;
//    if constexpr(TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
//    {
//        SurfPrimIdList sortedPrimIds = primIds;
//        std::sort(sortedPrimIds);
//        auto result = primBatchMapping.emplace(sortedPrimIds,
//                                               concreteAccelCounter);
//        // Get ready for next
//        if(result.second) concreteAccelCounter++;
//        concAccelIndex = result.first->second;
//    }
//    else concreteAccelCounter++;

//    surfaceIdCounter++;
//    surfMapping.emplace(surfaceIdCounter, concAccelIndex);
//}

//// Commit the accelerators return world space AABBs
//Span<AABB3> CommitReservations(const std::vector<BaseAcceleratorLeaf>& baseLeafs,
//                               const AcceleratorWorkI& accWork) override;
//{
//    accelAABBs.reserve(baseLeafs.size());
//    perAccelLeafs.reserve(baseLeafs.size());

//    if constexpr(TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
//    {
//        // Create Identity transform work
//        auto tg = IdentityTransformGroup{};
//        const auto& ag = decltype(this);
//        AcceleratorWork<decltype(this), IdentityTransformGroup> work(ag, tg);

//        std::unordered_map<uint32_t, Span<AcceleratorLeaf>> constructedAccels;
//        assert(primBatchMapping.empty());
//        std::array<PrimRange, MaxPrimBatchPerSurface> a;

//        auto it = constructedAccels.find(primBatchMapping.at());
//        for(const BaseAcceleratorLeaf& l : baseLeafs)
//        {
//            uint32_t accId = l.accelId.FetchIndexPortion();
//            uint32_t instanceId = surfMapping.at(accId);
//            auto r = constructedAccels.emplace(instanceId);
//            if(r.second)
//            {
//                MRAY_LOG("{:s}: Constructing Accelerator {}", TypeName(),
//                         ...);
//                ConstructAccelerator(a, indentityWork);
//            }
//            else
//            {
//                MRAY_LOG("{:s}: Reusing Accelerator for {}", TypeName(),
//                         ...);
//                ReferAccelerator(a, r.first->second);
//            }
//            AABB3 aabb = GenerateAABB(r.first->second, l.transformKey, accWork);
//            result.push_back(aabb);

//        }
//    }
//    else
//    {
//        MRAY_LOG("{:s}: Constructing Accelerator {}", TypeName(),
//                 ...);
//        ConstructAccelerator(a, accWork);
//        AABB3 aabb = GenerateAABB(r.first->second, l.transformKey, accWork);
//        result.push_back(aabb);
//        perAccelLeafs.push_back(leafs);
//    }

//    // Memcopy to a span

//}