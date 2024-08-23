#pragma once

namespace LBVHAccelDetail
{

std::vector<AABB3> MulitBuildLBVH(const GPUQueue& queue)
{
    std::vector<AABB3> hResultingAABBs;

    //// Allocate temp memory
    //// Construction of LBVH requires sorting (of MortonCodes)
    //// calculate that, we also need to reduce the AABBs beforehand,
    //// to find and optimal Morton Code delta
    //using namespace DeviceAlgorithms;
    ////SegmentedRadixSort

    //size_t sortMemSize = RadixSortTMSize<true, uint64_t, uint32_t>(instanceCount);
    //size_t reduceMemSize = ReduceTMSize<AABB3>(dLeafAABBs.size());
    //size_t tempMemSize = std::max(sortMemSize, reduceMemSize);

    //// LBVH requires a special care when leaves have duplicate morton codes,
    //// We skip that and use large morton code (64-bit) and hope for the best.
    //// This is not optimal, but BVH is here for completeness sake.
    ////
    //// Reduce the given AABBs
    //// Cheeckly utilize stack mem as temp mem
    //Span<AABB3>     dSceneAABB;
    //Span<Byte>      dTempMem;
    //std::array<Span<uint32_t>, 2>   dIndices;
    //std::array<Span<uint64_t>, 2>   dLeafMortonCodes;
    //MemAlloc::AllocateMultiData(std::tie(dTempMem,
    //                                     dLeafMortonCodes[0], dLeafMortonCodes[1],
    //                                     dIndices[0], dIndices[1],
    //                                     dSceneAABB),
    //                            stackMem,
    //                            {tempMemSize, instanceCount,
    //                             instanceCount, instanceCount, 1});

    //const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    //Reduce(Span<AABB3, 1>(dSceneAABB), dTempMem,
    //       ToConstSpan(dLeafAABBs),
    //       AABB3::Negative(),
    //       queue, UnionAABB3Functor());

    //// Might aswell get the final AABB in the meantime
    //AABB3 hReducedAABB = AABB3::Negative();
    //queue.MemcpyAsync(Span<AABB3>(&hSceneAABB, 1), ToConstSpan(dSceneAABB));

    //// Initial index list
    //Iota(dIndices[0], 0u, queue);

    //using namespace std::string_view_literals;
    //queue.IssueSaturatingKernel<KCGenMortonCode>
    //(
    //    "(A)BVHGenerateMortonCodes"sv,
    //    KernelIssueParams{.workCount = uint32_t(instanceCount)},
    //    dLeafMortonCodes[0],
    //    dLeafAABBs,
    //    Span<const AABB3, 1>(dSceneAABB)
    //);
    //// TODO: Can we estimate the bit count?
    //// Do full bit sort
    //uint32_t sortedIndex = RadixSort<true, uint64_t, uint32_t>
    //(
    //    dLeafMortonCodes, dIndices,
    //    dTempMem, queue
    //);
    //uint32_t otherIndex = (sortedIndex == 0) ? 1 : 0;

    //// Check if we have duplicate morton code
    //Span<uint32_t> dDuplicateCheckBuffer = dIndices[otherIndex].subspan(0, instanceCount - 1);
    //Span<uint32_t> dDuplicateCheckResult = dIndices[otherIndex].subspan(instanceCount - 1, 1);
    //AdjacentDifference(dDuplicateCheckBuffer,
    //                   ToConstSpan(dLeafMortonCodes[sortedIndex]), queue,
    //                   MortonDiffFunctor());
    //Reduce(Span<uint32_t, 1>(dDuplicateCheckResult),
    //       dTempMem, ToConstSpan(dDuplicateCheckBuffer),
    //       0u, queue, std::plus<uint32_t>{});

    //// Do not synchronize the host here,
    //// throw the error later
    //uint32_t hDuplicateCheckResult;
    //queue.MemcpyAsync(Span<uint32_t>(&hDuplicateCheckResult, 1),
    //                  ToConstSpan(dDuplicateCheckResult));


    //// Now we have a valid morton code list


    // Wait for "hReducedAABB" and the temp device memory
    // and All Done!
    queue.Barrier().Wait();
    //if(hDuplicateCheckResult != 0)
    //{
    //    throw MRayError("AABBs on Base LBVH is too close together, "
    //                    "unable to construct accelerator!");
    //}
    return hResultingAABBs;
}

template <class IntersectFunc>
MRAY_GPU MRAY_GPU_INLINE
uint32_t TraverseLBVH(BitStack& bitStack,
                      // Inputs
                      Vector2 tMinMax,
                      const Ray& ray,
                      Span<const LBVHNode> nodes,
                      bool breakOnFirstHit,
                      uint32_t initialNodeIndex,
                      IntersectFunc&& Func)
{
    const LBVHNode* nodesPtr = nodes.data();

    ChildIndex nodeIndex(0);
    const LBVHNode* currentNode = nodesPtr + initialNodeIndex;
    while(bitStack.Depth() <= BitStack::MAX_DEPTH)
    {
        // SpecialCase: We are on leaf node, check primitive intersection
        // and pop back to parent
        if(nodeIndex.FetchBatchPortion() == IS_LEAF)
        {
            uint32_t leafIndex = nodeIndex.FetchIndexPortion();
            bool doIntersect = Func(tMinMax, leafIndex);
            if(breakOnFirstHit && doIntersect) break;

            bitStack.MarkAsTraversed();
            bitStack.Ascend();
            nodeIndex = ChildIndex(0);
            continue;
        }
        // Determine traverse information
        BitStack::TraverseState traverseState = bitStack.CurrentState();
        // Fresh entry, we never checked this node,
        // If intersects decsnd
        if(traverseState == BitStack::FIRST_ENTRY &&
           ray.IntersectsAABB(currentNode->aabb.Min(),
                              currentNode->aabb.Max(),
                              tMinMax))
        {
            nodeIndex = ChildIndex(currentNode->leftIndex);
            if(nodeIndex.FetchBatchPortion() != IS_LEAF)
                currentNode = nodesPtr + nodeIndex.FetchIndexPortion();
            bitStack.Descend();
        }
        // Nothing to see here, go up
        else if(traverseState == BitStack::FIRST_ENTRY)
        {
            currentNode = nodesPtr + currentNode->parentIndex;
            bitStack.MarkAsTraversed();
            bitStack.Ascend();
        }
        // Coming from left (we traverse left first, then right)
        // and directly going to right
        else if(traverseState == BitStack::U_TURN)
        {
            nodeIndex = currentNode->rightIndex;
            if(nodeIndex.FetchBatchPortion() != IS_LEAF)
                currentNode = nodesPtr + nodeIndex.FetchIndexPortion();
            bitStack.Descend();
            break;
        }
        // Just go up (state is 0b10, 0b11 should not be possible)
        else
        {
            currentNode = nodesPtr + currentNode->parentIndex;
            bitStack.WipeLowerBits();
            bitStack.Ascend();
            break;
        }
    }
    return std::distance(nodesPtr, currentNode);
}

MRAY_GPU MRAY_GPU_INLINE
BitStack::BitStack(uint32_t initialState, uint32_t initialDepth)
    : stack(initialState)
    , depth(initialDepth)
{}

MRAY_GPU MRAY_GPU_INLINE
void BitStack::WipeLowerBits()
{
    uint32_t mask = std::numeric_limits<uint32_t>::max() << uint32_t(depth - 1);
    stack &= mask;
}

MRAY_GPU MRAY_GPU_INLINE
typename BitStack::TraverseState BitStack::CurrentState() const
{
    return TraverseState((stack >> (depth - 2)) & 0x3u);
}

MRAY_GPU MRAY_GPU_INLINE
void BitStack::MarkAsTraversed()
{
    stack += (1u << (depth - 1));
}

MRAY_GPU MRAY_GPU_INLINE
void BitStack::Descend()
{
    depth--;
}

MRAY_GPU MRAY_GPU_INLINE
void BitStack::Ascend()
{
    depth++;
}

MRAY_GPU MRAY_GPU_INLINE
uint32_t BitStack::Depth() const
{
    return depth;
}

template<uint32_t SBits, uint32_t DBits>
MRAY_GPU MRAY_GPU_INLINE
uint32_t BitStack::CompressState() const
{
    return Bit::Compose<SBits, DBits>(stack, depth);
}

template<PrimitiveGroupC PG, TransformGroupC TG>
template<auto GenerateTransformContext>
MRAY_GPU MRAY_GPU_INLINE
OptionalHitR<PG> AcceleratorLBVH<PG, TG>::IntersectionCheck(const Ray& ray,
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
MRAY_GPU MRAY_GPU_INLINE
AcceleratorLBVH<PG, TG>::AcceleratorLBVH(const TransDataSoA& tSoA,
                                         const PrimDataSoA& pSoA,
                                         const DataSoA& dataSoA,
                                         AcceleratorKey aId)
    : primRanges(dataSoA.dPrimitiveRanges[aId.FetchIndexPortion()])
    , cullFaceFlags(dataSoA.dCullFace[aId.FetchIndexPortion()])
    , alphaMaps(dataSoA.dAlphaMaps[aId.FetchIndexPortion()])
    , lmKeys(dataSoA.dLightOrMatKeys[aId.FetchIndexPortion()])
    , leafs(dataSoA.dLeafs[aId.FetchIndexPortion()])
    , nodes(dataSoA.dNodes[aId.FetchIndexPortion()])
    , rootNodeIndex(dataSoA.dInstanceRootNodeIndices[aId.FetchIndexPortion()])
    , transformKey(dataSoA.dInstanceTransforms[aId.FetchIndexPortion()])
    , transformSoA(tSoA)
    , primitiveSoA(pSoA)
{}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
TransformKey AcceleratorLBVH<PG, TG>::TransformKey() const
{
    return transformKey;
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
OptionalHitR<PG> AcceleratorLBVH<PG, TG>::ClosestHit(BackupRNG& rng,
                                                     const Ray& ray,
                                                     const Vector2& tMinMax) const
{
    BitStack bitStack;
    OptionalHitR<PG> result = std::nullopt;
    TraverseLBVH(bitStack, tMinMax, ray, nodes, false,
                 rootNodeIndex,
    [&](Vector2& tMM, uint32_t leafIndex)
    {
        PrimitiveKey primKey = leafs[leafIndex];
        result = IntersectionCheck(ray, tMM, rng.NextFloat(), primKey);
        if(result) tMM[1] = min(tMM[1], result->t);
        return result.has_value();
    });
    return result;
}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
OptionalHitR<PG> AcceleratorLBVH<PG, TG>::FirstHit(BackupRNG& rng,
                                                   const Ray& ray,
                                                   const Vector2& tMinMax) const
{
    BitStack bitStack;
    OptionalHitR<PG> result = std::nullopt;
    TraverseLBVH(bitStack, tMinMax, ray, nodes, true,
                 rootNodeIndex,
    [&](Vector2& tMM, uint32_t leafIndex)
    {
        PrimitiveKey primKey = leafs[leafIndex];
        result = IntersectionCheck(ray, tMM, rng.NextFloat(), primKey);
        return result.has_value();
    });
}

}

template<PrimitiveGroupC PG>
std::string_view AcceleratorGroupLBVH<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = AccelGroupTypeName(BaseAcceleratorLBVH::TypeName(),
                                                PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupLBVH<PG>::AcceleratorGroupLBVH(uint32_t accelGroupId,
                                                   BS::thread_pool& tp,
                                                   const GPUSystem& sys,
                                                   const GenericGroupPrimitiveT& pg,
                                                   const AccelWorkGenMap& globalWorkMap)
    : Base(accelGroupId, tp, sys, pg, globalWorkMap)
    , mem(sys.AllGPUs(), 2_MiB, 32_MiB)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupLBVH<PG>::Construct(AccelGroupConstructParams p,
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
                                tempMem,
                                {this->concreteLeafRanges.size(),
                                 ppResult.concretePrimRanges.size()});
    assert(ppResult.concretePrimRanges.size() ==
           this->concreteLeafRanges.size());

    // Calculate all node size
    // Thankfully LBVH has implicit node count which is "leafCount - 1"
    // So using "leafCount" as conservative estimate
    uint32_t totalNodeCount = this->concreteLeafRanges.back()[1];
    // Copy these host vectors to GPU
    MemAlloc::AllocateMultiData(std::tie(dCullFaceFlags,
                                         dAlphaMaps,
                                         dLightOrMatKeys,
                                         dPrimitiveRanges,
                                         dTransformKeys,
                                         dRootNodeIndices,
                                         dLeafs,
                                         dNodes,
                                         dAllLeafs,
                                         dAllNodes),
                                mem,
                                {this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->concreteLeafRanges.back()[1],
                                 totalNodeCount});
    // Generate offset spans
    using PrimKeySpanList = std::vector<Span<PrimitiveKey>>;
    using NodeSpanList = std::vector<Span<LBVHNode>>;
    PrimKeySpanList hInstanceLeafs = this->CreateInstanceLeafSubspans(dAllLeafs);
    NodeSpanList hInstanceNodes = this->CreateInstanceLeafSubspans(dAllNodes);

    // Actual memcpy
    Span<CullFaceFlagArray>     hSpanCullFaceFlags(ppResult.surfData.cullFaceFlags);
    Span<AlphaMapArray>         hSpanAlphaMaps(ppResult.surfData.alphaMaps);
    Span<LightOrMatKeyArray>    hSpanLMKeys(ppResult.surfData.lightOrMatKeys);
    Span<PrimRangeArray>        hSpanPrimitiveRanges(ppResult.surfData.primRanges);
    Span<TransformKey>          hSpanTransformKeys(ppResult.surfData.transformKeys);
    Span<Span<PrimitiveKey>>    hSpanLeafs(hInstanceLeafs);
    Span<Span<LBVHNode>>        hSpanNodes(hInstanceNodes);
    queue.MemcpyAsync(dCullFaceFlags,   ToConstSpan(hSpanCullFaceFlags));
    queue.MemcpyAsync(dAlphaMaps,       ToConstSpan(hSpanAlphaMaps));
    queue.MemcpyAsync(dLightOrMatKeys,  ToConstSpan(hSpanLMKeys));
    queue.MemcpyAsync(dPrimitiveRanges, ToConstSpan(hSpanPrimitiveRanges));
    queue.MemcpyAsync(dTransformKeys,   ToConstSpan(hSpanTransformKeys));
    queue.MemcpyAsync(dLeafs,           ToConstSpan(hSpanLeafs));
    queue.MemcpyAsync(dNodes,           ToConstSpan(hSpanNodes));

    // Copy Ids to the leaf buffer
    auto hConcreteLeafRanges = Span<const Vector2ui>(this->concreteLeafRanges.begin(),
                                                     this->concreteLeafRanges.end());
    auto hConcretePrimRanges = Span<const PrimRangeArray>(ppResult.concretePrimRanges.cbegin(),
                                                          ppResult.concretePrimRanges.cend());
    queue.MemcpyAsync(dConcreteLeafRanges, hConcreteLeafRanges);
    queue.MemcpyAsync(dConcretePrimRanges, hConcretePrimRanges);

    // Dedicate a block for each
    // concrete accelerator for copy
    uint32_t blockCount = queue.RecommendedBlockCountDevice(&KCGeneratePrimitiveKeys,
                                                            StaticThreadPerBlock1D(),
                                                            0);
    using namespace std::string_literals;
    static const auto KernelName = "KCGeneratePrimitiveKeys-"s + std::string(TypeName());
    queue.IssueExactKernel<KCGeneratePrimitiveKeys>
    (
        KernelName,
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

    // Easy part is done
    // We need to actually construct this thing now,
    // .....................................
    // TODO:
    // .....................................
    //
    // 1. Do segmented reduce to find centers and AABBs of the primitives
    // 2. Calculate temporary morton codes of each primitive using the AABBs + centers
    // 3. Do segmented radix sort to find the actual layout of the LBVH
    // 4. Call dynamic kernel to bottom up generate the hierarchy
    // 5. Actually create the BVH nodes
    // 6. Bottom-up reduce the BVH's
    //
    // And it should be done
    // Since we need to do these on base accelerator as well a generic impl should
    // be considered.

    // We have temp memory + async memcopies,
    // we need to wait here.
    queue.Barrier().Wait();
}

template<PrimitiveGroupC PG>
typename AcceleratorGroupLBVH<PG>::DataSoA
AcceleratorGroupLBVH<PG>::SoA() const
{
    return data;
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupLBVH<PG>::GPUMemoryUsage() const
{
    return mem.Size();
}

template<PrimitiveGroupC PG>
void AcceleratorGroupLBVH<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
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
void AcceleratorGroupLBVH<PG>::CastLocalRays(// Output
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