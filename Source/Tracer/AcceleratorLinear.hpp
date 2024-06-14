#pragma once

namespace LinearAccelDetail
{



template<PrimitiveGroupC PG, TransformGroupC TG>
template<auto GenerateTransformContext>
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
                                                   BS::thread_pool& tp,
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
    // Before the allocation hickup, allocate the temp
    // memory as well.
    DeviceLocalMemory tempMem(*queue.Device());
    // Allocate concrete leaf ranges for processing
    // Copy device
    Span<Vector2ui> dConcreteLeafRanges;
    Span<PrimRangeArray> dConcretePrimRanges;
    MemAlloc::AllocateMultiData(std::tie(dConcreteLeafRanges,
                                         dConcretePrimRanges),
                                mem,
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
    using PrimKeySpanList = std::vector<Span<PrimitiveKey>>;
    PrimKeySpanList hInstanceLeafs = this->CreateInstanceLeafSubspans(dAllLeafs);

    // Actual memcpy
    Span<CullFaceFlagArray>     hSpanCullFaceFlags(ppResult.surfData.cullFaceFlags);
    Span<AlphaMapArray>         hSpanAlphaMaps(ppResult.surfData.alphaMaps);
    Span<LightOrMatKeyArray>    hSpanLMKeys(ppResult.surfData.lightOrMatKeys);
    Span<PrimRangeArray>        hSpanPrimitiveRanges(ppResult.surfData.primRanges);
    Span<TransformKey>          hSpanTransformKeys(ppResult.surfData.transformKeys);
    Span<Span<PrimitiveKey>>    hSpanLeafs(hInstanceLeafs);
    queue.MemcpyAsync(dCullFaceFlags,   ToConstSpan(hSpanCullFaceFlags));
    queue.MemcpyAsync(dAlphaMaps,       ToConstSpan(hSpanAlphaMaps));
    queue.MemcpyAsync(dLightOrMatKeys,  ToConstSpan(hSpanLMKeys));
    queue.MemcpyAsync(dPrimitiveRanges, ToConstSpan(hSpanPrimitiveRanges));
    queue.MemcpyAsync(dTransformKeys,   ToConstSpan(hSpanTransformKeys));
    queue.MemcpyAsync(dLeafs,           ToConstSpan(hSpanLeafs));


    // Copy Ids to the leaf buffer
    auto hConcreteLeafRanges = Span<const Vector2ui>(this->concreteLeafRanges.begin(),
                                                     this->concreteLeafRanges.end());
    auto hConcretePrimRanges = Span<const PrimRangeArray>(ppResult.concretePrimRanges.cbegin(),
                                                          ppResult.concretePrimRanges.cend());
    queue.MemcpyAsync(dConcreteLeafRanges, hConcreteLeafRanges);
    queue.MemcpyAsync(dConcretePrimRanges, hConcretePrimRanges);

    // Dedicate a block for each
    // concrete accelerator for copy
    uint32_t blockCount = queue.SMCount() *
        GPUQueue::RecommendedBlockCountPerSM(&KCGeneratePrimitiveKeys,
                                             StaticThreadPerBlock1D(),
                                             0);
    using namespace std::string_view_literals;
    queue.IssueExactKernel<KCGeneratePrimitiveKeys>
    (
        "KCGeneratePrimitiveKeys"sv,
        KernelExactIssueParams
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
    // We have temp memory + async memcopies
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
    if constexpr(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
    {

        using namespace DeviceAlgorithms;
        size_t tmSize = SegmentedTransformReduceTMSize<AABB3, PrimitiveKey>(this->concreteLeafRanges.size());
        Span<uint32_t> dConcreteIndicesOfInstances;
        Span<AABB3> dConcreteAABBs;
        Span<uint32_t> dConcreteLeafOffsets;
        Span<Byte> dTransformSegReduceTM;
        MemAlloc::AllocateMultiData(std::tie(dConcreteIndicesOfInstances,
                                             dConcreteAABBs,
                                             dConcreteLeafOffsets,
                                             dTransformSegReduceTM),
                                    tempMem,
                                    {totalInstanceCount, concreteAccelCount,
                                     concreteAccelCount + 1,
                                     tmSize});
        Span<const uint32_t> hConcreteIndicesOfInstances(this->concreteIndicesOfInstances.data(),
                                                         this->concreteIndicesOfInstances.size());
        Span<const Vector2ui> hConcreteLeafRanges(this->concreteLeafRanges.data(),
                                                  this->concreteLeafRanges.size());
        // Normal copy to GPU
        queue.MemcpyAsync(dConcreteIndicesOfInstances, hConcreteIndicesOfInstances);

        // We need to copy the Vector2ui [(0, n_0), [n_0, n_1), ..., [n_{m-1}, n_m)]
        // As [n_0, n_1, ..., n_{m-1}, n_m]
        // This is technically UB maybe?
        // But it is hard to recognize by the compiler maybe? Dunno
        // Do a sanity check at least...
        static_assert(sizeof(Vector2ui) == 2 * sizeof(typename Vector2ui::InnerType));
        Span<const uint32_t> hConcreteLeafRangesInt(hConcreteLeafRanges.data()->AsArray().data(),
                                                    hConcreteLeafRanges.size() * Vector2ui::Dims);

        // Memset the first element to zero
        queue.MemsetAsync(dConcreteLeafOffsets.subspan(0, 1), 0x00);
        queue.MemcpyAsyncStrided(dConcreteLeafOffsets.subspan(1), 0,
                                 hConcreteLeafRangesInt.subspan(1), sizeof(Vector2ui));

        typename PrimitiveGroup::DataSoA pData = this->pg.SoA();
        SegmentedTransformReduce<AABB3, PrimitiveKey>
        (
            dConcreteAABBs,
            dTransformSegReduceTM,
            ToConstSpan(dAllLeafs),
            dConcreteLeafOffsets,
            AABB3::Negative(),
            queue,
            UnionAABB3Functor(),
            [pData] MRAY_HYBRID(PrimitiveKey k)
            {
                using Prim = typename PrimitiveGroup:: template Primitive<>;
                Prim prim(TransformContextIdentity{}, pData, k);
                AABB3 aabb = prim.GetAABB();
                return aabb;
            }
        );
        // Now, copy (and transform) concreteAABBs (which are on local space)
        // to actual accelerator instance aabb's (after transform these will be
        // in world space)
        for(const auto& kv : this->workInstances)
        {
            uint32_t index = kv.first;
            const AccelWorkPtr& workPtr = kv.second;
            size_t size = (this->workInstanceOffsets[index + 1] -
                           this->workInstanceOffsets[index]);
            Span<const uint32_t> dLocalIndices = dConcreteIndicesOfInstances.subspan(this->workInstanceOffsets[index], size);
            Span<const TransformKey> dLocalTKeys = dTransformKeys.subspan(this->workInstanceOffsets[index], size);
            Span<AABB3> dLocalWriteRegion = aabbWriteRegion.subspan(this->workInstanceOffsets[index], size);

            workPtr->TransformLocallyConstantAABBs(dLocalWriteRegion,
                                                   dConcreteAABBs,
                                                   dLocalIndices,
                                                   dLocalTKeys,
                                                   queue);
        }
        //  Don't forget to wait for temp memory!
        queue.Barrier().Wait();
    }
    else
    {
        throw MRayError("{}: PER_PRIM_TRANSFORM Accel Construct not yet implemented",
                        TypeName());
    }

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