#pragma once

#include "AcceleratorLinear.h"
#include "Tracer/AcceleratorWork.h"

namespace LinearAccelDetail
{

template<PrimitiveGroupC PG, TransformGroupC TG>
template<auto GenerateTransformContext>
MRAY_GPU MRAY_GPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::IntersectionCheck(const Ray& ray,
                                                              const Vector2f& tMinMax,
                                                              Float xi,
                                                              const PrimitiveKey& primKey) const
{
    auto IsInRange = [tMinMax](Float newT) -> bool
    {
        return ((newT >= tMinMax[0]) && (newT < tMinMax[1]));
    };

    using Primitive = typename PG::template Primitive<TransformContextIdentity>;
    using Intersection = IntersectionT<PrimHit>;
    using enum PrimTransformType;

    Ray transformedRay;
    if constexpr(PG::TransformLogic == PER_PRIMITIVE_TRANSFORM)
    {
        // Primitive has per primitive transform (skinned mesh maybe).
        // we need to transform using primitive's data
        // Create transform context and transform the ray.
        using TransContext = typename PrimTransformContextType<PG, TG>::Result;
        TransContext tContext = GenerateTransformContext(transformSoA, primitiveSoA,
                                                         transformKey, primKey);
        transformedRay = tContext.InvApply(ray);
    }
    else
    {
        // Otherwise, the transform is constant local transform,
        // ray is already transformed to local space by the caller
        // so do not transform
        transformedRay = ray;
    }

    // Construct the primitive
    Primitive prim = Primitive(TransformContextIdentity{}, primitiveSoA, primKey);
    // Find the batch index for flags, alpha maps
    uint32_t index = FindPrimBatchIndex(primRanges, primKey);
    // Actual intersection finally!
    Optional<Intersection> intersection = prim.Intersects(transformedRay,
                                                          cullFaceFlags[index]);
    // Intersection decisions
    if(!intersection) return std::nullopt;
    if(!IsInRange(intersection->t)) return std::nullopt;

    Optional<AlphaMap> alphaMap = alphaMaps[index];
    if(alphaMap)
    {
        const auto& alphaMapV = alphaMap.value();
        // This has alpha map check it
        Vector2 uv = prim.SurfaceParametrization(intersection.value().hit);
        Float alpha = alphaMapV(uv);
        // Stochastic alpha culling
        if(xi >= alpha) return std::nullopt;
    }

    // It is a hit! Update
    return HitResult
    {
        .hit            = intersection.value().hit,
        .t              = intersection.value().t,
        .primitiveKey   = primKey,
        .lmKey          = lmKeys[index]
    };
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
AcceleratorLinear<PG, TG>::AcceleratorLinear(const TransDataSoA& tSoA,
                                             const PrimDataSoA& pSoA,
                                             const DataSoA& dataSoA,
                                             AcceleratorKey aId)
    : primRanges(dataSoA.dPrimitiveRanges[aId.FetchIndexPortion()])
    , cullFaceFlags(dataSoA.dCullFace[aId.FetchIndexPortion()])
    , alphaMaps(dataSoA.dAlphaMaps[aId.FetchIndexPortion()])
    , lmKeys(dataSoA.dLightOrMatKeys[aId.FetchIndexPortion()])
    , leafs(ToConstSpan(dataSoA.dLeafs[aId.FetchIndexPortion()]))
    , transformKey(dataSoA.dInstanceTransforms[aId.FetchIndexPortion()])
    , transformSoA(tSoA)
    , primitiveSoA(pSoA)
{}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
TransformKey AcceleratorLinear<PG, TG>::GetTransformKey() const
{
    return transformKey;
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::ClosestHit(BackupRNG& rng,
                                                       const Ray& ray,
                                                       const Vector2& tMinMax) const
{
    Vector2 tMM = tMinMax;
    // Linear search over the array
    OptionalHitR<PG> result = std::nullopt;
    for(const PrimitiveKey pKeys : leafs)
    {
        auto check = IntersectionCheck(ray, tMM, rng.NextFloat(), pKeys);
        if(check.has_value() && check->t < tMM[1])
        {
            result = check;
            tMM[1] = check->t;
        }
    }
    return result;
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::FirstHit(BackupRNG& rng,
                                                     const Ray& ray,
                                                     const Vector2f& tMinMax) const
{
    // Linear search over the array
    OptionalHitR<PG> result = std::nullopt;
    for(const PrimitiveKey pKeys : leafs)
    {
        result = IntersectionCheck(ray, tMinMax, rng.NextFloat(), pKeys);
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
                                                   ThreadPool& tp,
                                                   const GPUSystem& sys,
                                                   const GenericGroupPrimitiveT& pg,
                                                   const AccelWorkGenMap& globalWorkMap)
    : Base(accelGroupId, tp, sys, pg, globalWorkMap)
    , mem(sys.AllGPUs(), 2_MiB, 32_MiB)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::Construct(AccelGroupConstructParams p,
                                           const GPUQueue& queue)
{
    PreprocessResult ppResult = this->PreprocessConstructionParams(p);
    // Before the allocation hiccup, allocate the temp
    // memory as well.
    DeviceLocalMemory tempMem(*queue.Device());
    // Allocate concrete leaf ranges for processing
    // Copy device
    Span<Vector2ui> dConcreteLeafRanges;
    Span<PrimRangeArray> dConcretePrimRanges;
    MemAlloc::AllocateMultiData(std::tie(dConcreteLeafRanges,
                                         dConcretePrimRanges),
                                tempMem,
                                {this->concreteLeafRanges.size(),
                                 ppResult.concretePrimRanges.size()});
    assert(ppResult.concretePrimRanges.size() ==
           this->concreteLeafRanges.size());
    // Copy these host vectors to GPU
    // For linear accelerator we only need these at GPU memory
    // additionally we will create instance's globalAABB
    MemAlloc::AllocateMultiData(std::tie(dCullFaceFlags,
                                         dAlphaMaps,
                                         dLightOrMatKeys,
                                         dPrimitiveRanges,
                                         dTransformKeys,
                                         dLeafs,
                                         dAllLeafs),
                                mem,
                                {this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->concreteLeafRanges.back()[1]});
    // Generate offset spans
    using PrimKeySpanList = std::vector<Span<const PrimitiveKey>>;
    PrimKeySpanList hInstanceLeafs = this->CreateInstanceSubspans(ToConstSpan(dAllLeafs),
                                                                  this->instanceLeafRanges);

    // Actual memcpy
    Span<CullFaceFlagArray>         hSpanCullFaceFlags(ppResult.surfData.cullFaceFlags);
    Span<AlphaMapArray>             hSpanAlphaMaps(ppResult.surfData.alphaMaps);
    Span<LightOrMatKeyArray>        hSpanLMKeys(ppResult.surfData.lightOrMatKeys);
    Span<PrimRangeArray>            hSpanPrimitiveRanges(ppResult.surfData.primRanges);
    Span<TransformKey>              hSpanTransformKeys(ppResult.surfData.transformKeys);
    Span<Span<const PrimitiveKey>>  hSpanLeafs(hInstanceLeafs);
    queue.MemcpyAsync(dCullFaceFlags,   ToConstSpan(hSpanCullFaceFlags));
    queue.MemcpyAsync(dAlphaMaps,       ToConstSpan(hSpanAlphaMaps));
    queue.MemcpyAsync(dLightOrMatKeys,  ToConstSpan(hSpanLMKeys));
    queue.MemcpyAsync(dPrimitiveRanges, ToConstSpan(hSpanPrimitiveRanges));
    queue.MemcpyAsync(dTransformKeys,   ToConstSpan(hSpanTransformKeys));
    queue.MemcpyAsync(dLeafs,           ToConstSpan(hSpanLeafs));

    // Copy Ids to the leaf buffer
    auto hConcreteLeafRanges = Span<const Vector2ui>(this->concreteLeafRanges);
    auto hConcretePrimRanges = Span<const PrimRangeArray>(ppResult.concretePrimRanges);
    queue.MemcpyAsync(dConcreteLeafRanges, hConcreteLeafRanges);
    queue.MemcpyAsync(dConcretePrimRanges, hConcretePrimRanges);

    // Dedicate a block for each
    // concrete accelerator for copy
    uint32_t blockCount = static_cast<uint32_t>(dConcreteLeafRanges.size());
    using namespace std::string_literals;
    static const auto KernelName = "KCGeneratePrimitiveKeys-"s + std::string(TypeName());
    queue.IssueBlockKernel<KCGeneratePrimitiveKeys>
    (
        KernelName,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = StaticThreadPerBlock1D()
        },
        // Output
        dAllLeafs,
        // Input
        dConcretePrimRanges,
        dConcreteLeafRanges,
        // Constant
        this->pg.GroupId()
    );

    data = DataSoA
    {
        .dCullFace              = ToConstSpan(dCullFaceFlags),
        .dAlphaMaps             = ToConstSpan(dAlphaMaps),
        .dLightOrMatKeys        = ToConstSpan(dLightOrMatKeys),
        .dPrimitiveRanges       = ToConstSpan(dPrimitiveRanges),
        .dInstanceTransforms    = ToConstSpan(dTransformKeys),
        .dLeafs                 = ToConstSpan(dLeafs)
    };


    // We have temp memory + async memcopies,
    // we need to wait here.
    queue.Barrier().Wait();
}

template<PrimitiveGroupC PG>
typename AcceleratorGroupLinear<PG>::DataSoA
AcceleratorGroupLinear<PG>::SoA() const
{
    return data;
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupLinear<PG>::GPUMemoryUsage() const
{
    return mem.Size();
}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                                           Span<AcceleratorKey> dKeyWriteRegion,
                                                           const GPUQueue& queue) const
{
    this->WriteInstanceKeysAndAABBsInternal(dAABBWriteRegion,
                                            dKeyWriteRegion,
                                            dAllLeafs,
                                            dTransformKeys,
                                            queue);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::CastLocalRays(// Output
                                               Span<HitKeyPack> dHitIds,
                                               Span<MetaHit> dHitParams,
                                               // I-O
                                               Span<BackupRNGState> dRNGStates,
                                               Span<RayGMem> dRays,
                                               // Input
                                               Span<const RayIndex> dRayIndices,
                                               Span<const CommonKey> dAccelKeys,
                                               // Constants
                                               CommonKey workId,
                                               const GPUQueue& queue)
{
    CommonKey localWorkId = workId - this->globalWorkIdToLocalOffset;
    const auto& workOpt = this->workInstances.at(localWorkId);

    if(!workOpt)
        throw MRayError("{:s}:{:d}: Unable to find work for {:d}",
                        TypeName(), this->accelGroupId, workId);

    const auto& work = workOpt.value().get();
    work->CastLocalRays(// Output
                        dHitIds,
                        dHitParams,
                        // I-O
                        dRNGStates,
                        dRays,
                        //Input
                        dRayIndices,
                        dAccelKeys,
                        // Constants
                        queue);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::CastVisibilityRays(// Output
                                                    Bitspan<uint32_t> dIsVisibleBuffer,
                                                    // I-O
                                                    Span<BackupRNGState> dRNGStates,
                                                    // Input
                                                    Span<const RayGMem> dRays,
                                                    Span<const RayIndex> dRayIndices,
                                                    Span<const CommonKey> dAccelKeys,
                                                    // Constants
                                                    CommonKey workId,
                                                    const GPUQueue& queue)
{
    CommonKey localWorkId = workId - this->globalWorkIdToLocalOffset;
    const auto& workOpt = this->workInstances.at(localWorkId);

    if(!workOpt)
        throw MRayError("{:s}:{:d}: Unable to find work for {:d}",
                        TypeName(), this->accelGroupId, workId);

    const auto& work = workOpt.value().get();
    work->CastVisibilityRays(// Output
                             dIsVisibleBuffer,
                             // I-O
                             dRNGStates,
                             //Input
                             dRays,
                             dRayIndices,
                             dAccelKeys,
                             // Constants
                             queue);
}