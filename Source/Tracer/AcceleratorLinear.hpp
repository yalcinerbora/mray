#pragma once

namespace LinearAccelDetail
{



template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
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
                                                 transformKey, primKey);

        transformedRay = transformContext.InvApply(ray);
    }
    else
    {
        // Otherwise, the transform is constant local transform,
        // ray is already transformed to local space
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
        Float alpha = alphaMapV(uv).value();
        // Stochastic alpha culling
        if(xi > alpha) return std::nullopt;
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
MRAY_HYBRID MRAY_CGPU_INLINE
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
MRAY_HYBRID MRAY_CGPU_INLINE
TransformKey AcceleratorLinear<PG, TG>::TransformKey() const
{
    return transformKey;
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_HYBRID MRAY_CGPU_INLINE
OptionalHitR<PG> AcceleratorLinear<PG, TG>::ClosestHit(BackupRNG& rng,
                                                       const Ray& ray,
                                                       const Vector2& tMinMax) const
{
    Vector2 tMM = tMinMax;
    // Linear search over the array
    OptionalHitR<PG> result = std::nullopt;
    for(const PrimitiveKey pKeys : leafs)
    {
        result = IntersectionCheck(ray, tMM, rng.NextFloat(), pKeys);
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
                                                   BS::thread_pool& tp, GPUSystem& sys,
                                                   const GenericGroupPrimitiveT& pg,
                                                   const AccelWorkGenMap& globalWorkMap)
    : Base(accelGroupId, tp, sys, pg, globalWorkMap)
    , mem(sys.AllGPUs(), 2_MiB, 32_MiB)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::Construct(AccelGroupConstructParams p,
                                           const GPUQueue& queue)
{
    LinearizedSurfaceData linearizedData = this->PreprocessConstructionParams(p);

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
    using PrimKeySpanList = std::vector<Span<PrimitiveKey>>;
    PrimKeySpanList hInstanceLeafs = this->CreateInstanceLeafSubspans(dAllLeafs);

    // Actual memcpy
    Span<CullFaceFlagArray>     hSpanCullFaceFlags(linearizedData.cullFaceFlags);
    Span<AlphaMapArray>         hSpanAlphaMaps(linearizedData.alphaMaps);
    Span<LightOrMatKeyArray>    hSpanLMKeys(linearizedData.lightOrMatKeys);
    Span<PrimRangeArray>        hSpanPrimitiveRanges(linearizedData.primRanges);
    Span<TransformKey>          hSpanTransformKeys(linearizedData.transformKeys);
    Span<Span<PrimitiveKey>>    hSpanLeafs(hInstanceLeafs);
    queue.MemcpyAsync(dCullFaceFlags,   ToConstSpan(hSpanCullFaceFlags));
    queue.MemcpyAsync(dAlphaMaps,       ToConstSpan(hSpanAlphaMaps));
    queue.MemcpyAsync(dLightOrMatKeys,  ToConstSpan(hSpanLMKeys));
    queue.MemcpyAsync(dPrimitiveRanges, ToConstSpan(hSpanPrimitiveRanges));
    queue.MemcpyAsync(dTransformKeys,   ToConstSpan(hSpanTransformKeys));
    queue.MemcpyAsync(dLeafs,           ToConstSpan(hSpanLeafs));

    //Instantiate X
    //
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
void AcceleratorGroupLinear<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> aabbWriteRegion,
                                                           Span<AcceleratorKey> keyWriteRegion,
                                                           const GPUQueue& queue) const
{
    // Sanity Checks
    assert(aabbWriteRegion.size() == this->concreteIndicesOfInstances.size());
    assert(keyWriteRegion.size() == this->concreteIndicesOfInstances.size());

    size_t totalInstanceCount = this->concreteIndicesOfInstances.size();
    size_t concreteAccelCount = this->concreteLeafRanges.size();

    // We will use a temp memory here
    // TODO: Add stream ordered memory allocator stuff to the
    // Device abstraction side maybe?
    DeviceLocalMemory tempMem(*queue.Device());

    using enum PrimTransformType;
    if constexpr(PG::TransformLogic == PER_PRIMITIVE_TRANSFORM)
    {
        size_t maxReductionSize = std::transform_reduce(this->concreteLeafRanges.cbegin(),
                                                        this->concreteLeafRanges.cend(),
                                                        size_t(0),
        [](size_t l, size_t r) -> size_t
        {
            return std::max(l, r);
        },
        [](const Vector2ui range)
        {
            return range[1] - range[0];
        });

        using namespace DeviceAlgorithms;
        size_t tmSize = TransformReduceTMSize<AABB3, PrimitiveKey>(maxReductionSize);

        Span<Byte> dTransformReduceTempMem;
        Span<uint32_t> dConcreteIndicesOfInstances;
        Span<uint32_t> dConcreteAABBs;
        MemAlloc::AllocateMultiData(std::tie(dConcreteIndicesOfInstances,
                                             dConcreteAABBs,
                                             dTransformReduceTempMem),
                                    tempMem,
                                    {totalInstanceCount, concreteAccelCount,
                                     tmSize});


        ////queue.K

        //// Find AABBs of the concerete instances only
        //// Then transform it
        //for(size_t i = 0; i < this->concreteLeafRanges.size())
        //{

        //}
    }
    else
    {

    }

    //for()


    //dLeafs

}

template<PrimitiveGroupC PG>
void AcceleratorGroupLinear<PG>::CastLocalRays(// Output
                                               Span<HitKeyPack> dHitIds,
                                               Span<MetaHit> dHitParams,
                                               // I-O
                                               Span<BackupRNGState> rngStates,
                                               Span<RayGMem> dRays,
                                               // Input
                                               Span<const RayIndex> dRayIndices,
                                               Span<const CommonKey> dAccelKeys,
                                               // Constants
                                               uint32_t workId,
                                               const GPUQueue& queue)
{
    uint32_t localWorkId = workId - this->globalWorkIdToLocalOffset;
    const auto& workOpt = this->workInstances.at(localWorkId);

    if(!workOpt)
        throw MRayError("{:s}:{:d}: Unable to find work for {:d}",
                        TypeName(), this->accelGroupId, workId);

    const auto& work = workOpt.value().get();
    work->CastLocalRays(// Output
                        dHitIds,
                        dHitParams,
                        // I-O
                        rngStates,
                        dRays,
                        //Input
                        dRayIndices,
                        dAccelKeys,
                        // Constants
                        queue);
}