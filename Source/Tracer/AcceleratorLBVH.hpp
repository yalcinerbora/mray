#pragma once

namespace LBVHAccelDetail
{

template <uint32_t MAX_DEPTH, class IntersectFunc>
MRAY_HYBRID MRAY_GPU_INLINE
uint32_t TraverseLBVH(BitStack& bitStack,
                      // Traversal data
                      const Span<const LBVHNode>& nodes,
                      const Span<const LBVHBoundingBox>& bBoxes,
                      // Inputs
                      Vector2 tMinMax,
                      const Ray& ray,
                      uint32_t traverseStartIndex,
                      // Constants
                      IntersectFunc&& Func)
{
    assert(ChildIndex(traverseStartIndex).FetchBatchPortion() != IS_LEAF);
    const LBVHNode* nodesPtr = nodes.data();
    const LBVHBoundingBox* boxPtr = bBoxes.data();

    ChildIndex nodeIndex(0);
    const LBVHNode* currentNode = nodesPtr + traverseStartIndex;
    const LBVHBoundingBox* currentBBox = boxPtr + traverseStartIndex;
    while(bitStack.Depth() <= MAX_DEPTH)
    {
        // SpecialCase: We are on leaf node, check primitive intersection
        // and pop back to parent
        if(nodeIndex.FetchBatchPortion() == IS_LEAF)
        {
            // Rare edge case: We have single leaf; thus a single internal node
            // We enter right.
            // For every other LBVH; by construction, there are no null children
            // ever. TODO: We should not pay the price for this branch for other LBVH
            // change this later?
            if(nodeIndex == ChildIndex::InvalidKey())
                return std::numeric_limits<uint32_t>::max();

            uint32_t leafIndex = nodeIndex.FetchIndexPortion();
            bool breakTraversal = Func(tMinMax, leafIndex);
            bitStack.MarkAsTraversed();
            bitStack.Ascend();
            nodeIndex = ChildIndex(0);
            if(breakTraversal) break;
            // TODO: Probably not need this
            else continue;
        }
        // Determine traverse information
        BitStack::TraverseState traverseState = bitStack.CurrentState();
        // Fresh entry, we never checked this node,
        // If intersects descend
        if(traverseState == BitStack::FIRST_ENTRY &&
           ray.IntersectsAABB(Vector3(Span<const Float, 3>(currentBBox->min)),
                              Vector3(Span<const Float, 3>(currentBBox->max)),
                              tMinMax))
        {
            nodeIndex = ChildIndex(currentNode->leftIndex);
            if(nodeIndex.FetchBatchPortion() != IS_LEAF)
            {
                currentNode = nodesPtr + nodeIndex.FetchIndexPortion();
                currentBBox = boxPtr + nodeIndex.FetchIndexPortion();
            }
            bitStack.Descend();
        }
        // Nothing to see here, go up
        else if(traverseState == BitStack::FIRST_ENTRY)
        {
            uint32_t parentIndex = currentNode->parentIndex;
            currentNode = nodesPtr + parentIndex;
            currentBBox = boxPtr + parentIndex;
            bitStack.MarkAsTraversed();
            bitStack.Ascend();
        }
        // Coming from left (we traverse left first, then right)
        // and directly going to right
        else if(traverseState == BitStack::U_TURN)
        {
            nodeIndex = currentNode->rightIndex;
            if(nodeIndex.FetchBatchPortion() != IS_LEAF)
            {
                currentNode = nodesPtr + nodeIndex.FetchIndexPortion();
                currentBBox = boxPtr + nodeIndex.FetchIndexPortion();
            }
            bitStack.Descend();
            bitStack.MarkAsTraversed();
        }
        // Just go up (state is 0b10, 0b11 should not be possible)
        else
        {
            uint32_t parentIndex = currentNode->parentIndex;
            currentNode = nodesPtr + parentIndex;
            currentBBox = boxPtr + parentIndex;
            bitStack.WipeLowerBits();
            bitStack.Ascend();
        }
    }
    return uint32_t(std::distance(nodesPtr, currentNode));
}

#ifdef MRAY_GPU_BACKEND_CPU

template <uint32_t MAX_DEPTH, class IntersectFunc>
MRAY_HYBRID MRAY_GPU_INLINE
uint32_t TraverseLBVHStack(// I-O
                           BitStack&,
                           // Traversal data
                           const Span<const LBVHNode>& nodes,
                           const Span<const LBVHBoundingBox>& bBoxes,
                           // Inputs
                           Vector2 tMinMax,
                           const Ray& ray,
                           uint32_t traverseStartIndex,
                           // Constants
                           IntersectFunc&& Func)
{
    // With traditional recursive -> Stack implementation
    // There will be "DEPTH" amount of items in the stack
    // Assuming deepest branch's nodes all have children.
    // So we double the stack (+1 for good measure.
    using Stack = std::stack<ChildIndex, StaticVector<ChildIndex, MAX_DEPTH * 2u + 1>>;

    const LBVHNode* nodesPtr            = nodes.data();
    const LBVHBoundingBox* boxPtr       = bBoxes.data();
    const LBVHNode* currentNode         = nodesPtr + traverseStartIndex;
    const LBVHBoundingBox* currentBBox  = boxPtr + traverseStartIndex;
    //
    Stack nodeStack; nodeStack.push(ChildIndex(traverseStartIndex));
    while(!nodeStack.empty())
    {
        const ChildIndex nodeIndex = nodeStack.top();
        currentNode = nodesPtr + nodeIndex.FetchIndexPortion();
        currentBBox = boxPtr + nodeIndex.FetchIndexPortion();
        nodeStack.pop();

        // Rare edge case: We have single leaf; thus a single internal node
        // We enter right.
        // For every other LBVH; by construction, there are no null children
        // ever. TODO: We should not pay the price for this branch for other LBVH
        // change this later?
        if(nodeIndex == ChildIndex::InvalidKey())
            return std::numeric_limits<uint32_t>::max();

        // Case: Leaf
        if(nodeIndex.FetchBatchPortion() == IS_LEAF)
        {
            uint32_t leafIndex = nodeIndex.FetchIndexPortion();
            bool breakTraversal = Func(tMinMax, leafIndex);
            if(breakTraversal) break;
        }
        else if(ray.IntersectsAABB(Vector3(Span<const Float, 3>(currentBBox->min)),
                                   Vector3(Span<const Float, 3>(currentBBox->max)),
                                   tMinMax))
        {
            nodeStack.push(currentNode->leftIndex);
            nodeStack.push(currentNode->rightIndex);
        }
    }
    return uint32_t(std::distance(nodesPtr, currentNode));
}
#endif

MRAY_HYBRID MRAY_GPU_INLINE
BitStack::BitStack()
    : stack(0)
    , depth(MAX_DEPTH)
{}

MRAY_HYBRID MRAY_GPU_INLINE
BitStack::BitStack(uint64_t initialState, uint32_t initialDepth)
    : stack(initialState)
    , depth(initialDepth)
{}

MRAY_HYBRID MRAY_GPU_INLINE
void BitStack::WipeLowerBits()
{
    uint64_t mask = std::numeric_limits<uint64_t>::max() << (depth - 1);
    stack &= mask;
}

MRAY_HYBRID MRAY_GPU_INLINE
typename BitStack::TraverseState BitStack::CurrentState() const
{
    return TraverseState((stack >> (depth - 2)) & 0x3u);
}

MRAY_HYBRID MRAY_GPU_INLINE
void BitStack::MarkAsTraversed()
{
    stack += (uint64_t(1) << (depth - 1));
}

MRAY_HYBRID MRAY_GPU_INLINE
void BitStack::Descend()
{
    depth--;
}

MRAY_HYBRID MRAY_GPU_INLINE
void BitStack::Ascend()
{
    depth++;
}

MRAY_HYBRID MRAY_GPU_INLINE
uint32_t BitStack::Depth() const
{
    return depth;
}

template<uint32_t SBits, uint32_t DBits>
MRAY_HYBRID MRAY_GPU_INLINE
uint32_t BitStack::CompressState() const
{
    return uint32_t(Bit::Compose<SBits, DBits>(stack, depth));
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
    , boundingBoxes(dataSoA.dBoundingBoxes[aId.FetchIndexPortion()])
    , transformKey(dataSoA.dInstanceTransforms[aId.FetchIndexPortion()])
    , transformSoA(tSoA)
    , primitiveSoA(pSoA)
{}

template<PrimitiveGroupC PG, TransformGroupC TG>
MRAY_GPU MRAY_GPU_INLINE
TransformKey AcceleratorLBVH<PG, TG>::GetTransformKey() const
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
    //
    #ifdef MRAY_GPU_BACKEND_CPU
        TraverseLBVHStack<BitStack::MAX_DEPTH>
        (
            bitStack, nodes, boundingBoxes,
            tMinMax, ray, 0u,
            [&](Vector2& tMM, uint32_t leafIndex)
            {
                PrimitiveKey primKey = leafs[leafIndex];
                auto check = IntersectionCheck(ray, tMM, rng.NextFloat(), primKey);
                if(check.has_value() && check->t < tMM[1])
                {
                    result = check;
                    tMM[1] = check->t;
                }
                // Never terminate
                return false;
            }
        );
    #else
        TraverseLBVH<BitStack::MAX_DEPTH>
        (
            bitStack, nodes, boundingBoxes,
            tMinMax, ray, 0u,
            [&](Vector2& tMM, uint32_t leafIndex)
            {
                PrimitiveKey primKey = leafs[leafIndex];
                auto check = IntersectionCheck(ray, tMM, rng.NextFloat(), primKey);
                if(check.has_value() && check->t < tMM[1])
                {
                    result = check;
                    tMM[1] = check->t;
                }
                // Never terminate
                return false;
            }
        );
    #endif
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
    #ifdef MRAY_GPU_BACKEND_CPU
        TraverseLBVHStack<BitStack::MAX_DEPTH>
        (
            bitStack, nodes, boundingBoxes,
            tMinMax, ray, 0u,
            [&](Vector2& tMM, uint32_t leafIndex)
            {
                PrimitiveKey primKey = leafs[leafIndex];
                result = IntersectionCheck(ray, tMM, rng.NextFloat(), primKey);
                return result.has_value();
            }
        );
    #else
        TraverseLBVH<BitStack::MAX_DEPTH>
        (
            bitStack, nodes, boundingBoxes,
            tMinMax, ray, 0u,
            [&](Vector2& tMM, uint32_t leafIndex)
            {
                PrimitiveKey primKey = leafs[leafIndex];
                result = IntersectionCheck(ray, tMM, rng.NextFloat(), primKey);
                return result.has_value();
            }
        );
    #endif
    return result;
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
                                               ThreadPool& tp,
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
    // This can be derived from concrete leafranges.
    std::vector<Vector2ui> hConcreteNodeRangesVec;
    hConcreteNodeRangesVec.reserve(this->concreteLeafRanges.size());
    uint32_t nodeOffset = 0;
    for(const Vector2ui& leafRange : this->concreteLeafRanges)
    {
        uint32_t localLeafCount = leafRange[1] - leafRange[0];
        uint32_t localNodeCount = std::max(1u, localLeafCount - 1);
        hConcreteNodeRangesVec.push_back(Vector2ui(nodeOffset, nodeOffset + localNodeCount));
        nodeOffset += localNodeCount;
    }
    // Find the instance node ranges as well
    std::vector<Vector2ui> hInstanceNodeRangesVec;
    hInstanceNodeRangesVec.reserve(this->concreteIndicesOfInstances.size());
    for(uint32_t concreteIndex : this->concreteIndicesOfInstances)
        hInstanceNodeRangesVec.push_back(hConcreteNodeRangesVec[concreteIndex]);

    uint32_t totalLeafCount = this->concreteLeafRanges.back()[1];
    uint32_t totalNodeCount = hConcreteNodeRangesVec.back()[1];
    // Copy these host vectors to GPU
    MemAlloc::AllocateMultiData(std::tie(dCullFaceFlags,
                                         dAlphaMaps,
                                         dLightOrMatKeys,
                                         dPrimitiveRanges,
                                         dTransformKeys,
                                         dLeafs,
                                         dNodes,
                                         dNodeAABBs,
                                         dAllLeafs,
                                         dAllNodes,
                                         dAllNodeAABBs),
                                mem,
                                {this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 this->InstanceCount(), this->InstanceCount(),
                                 totalLeafCount, totalNodeCount, totalNodeCount});
    // Generate offset spans
    using PrimKeySpanList = std::vector<Span<const PrimitiveKey>>;
    using NodeSpanList = std::vector<Span<const LBVHNode>>;
    using NodeAABBSpanList = std::vector<Span<const LBVHBoundingBox>>;
    PrimKeySpanList hInstanceLeafs = this->CreateInstanceSubspans(ToConstSpan(dAllLeafs),
                                                                  this->instanceLeafRanges);
    NodeSpanList hInstanceNodes = this->CreateInstanceSubspans(ToConstSpan(dAllNodes),
                                                               hInstanceNodeRangesVec);
    NodeAABBSpanList hInstanceNodeAABBs = this->CreateInstanceSubspans(ToConstSpan(dAllNodeAABBs),
                                                                       hInstanceNodeRangesVec);

    // Actual memcpy
    Span<CullFaceFlagArray>             hSpanCullFaceFlags(ppResult.surfData.cullFaceFlags);
    Span<AlphaMapArray>                 hSpanAlphaMaps(ppResult.surfData.alphaMaps);
    Span<LightOrMatKeyArray>            hSpanLMKeys(ppResult.surfData.lightOrMatKeys);
    Span<PrimRangeArray>                hSpanPrimitiveRanges(ppResult.surfData.primRanges);
    Span<TransformKey>                  hSpanTransformKeys(ppResult.surfData.transformKeys);
    Span<Span<const PrimitiveKey>>      hSpanLeafs(hInstanceLeafs);
    Span<Span<const LBVHNode>>          hSpanNodes(hInstanceNodes);
    Span<Span<const LBVHBoundingBox>>   hSpanNodeBBoxes(hInstanceNodeAABBs);
    //
    queue.MemcpyAsync(dCullFaceFlags,   ToConstSpan(hSpanCullFaceFlags));
    queue.MemcpyAsync(dAlphaMaps,       ToConstSpan(hSpanAlphaMaps));
    queue.MemcpyAsync(dLightOrMatKeys,  ToConstSpan(hSpanLMKeys));
    queue.MemcpyAsync(dPrimitiveRanges, ToConstSpan(hSpanPrimitiveRanges));
    queue.MemcpyAsync(dTransformKeys,   ToConstSpan(hSpanTransformKeys));
    queue.MemcpyAsync(dLeafs,           ToConstSpan(hSpanLeafs));
    queue.MemcpyAsync(dNodes,           ToConstSpan(hSpanNodes));
    queue.MemcpyAsync(dNodeAABBs,       ToConstSpan(hSpanNodeBBoxes));

    // Copy Ids to the leaf buffer
    auto hConcreteLeafRanges = Span<const Vector2ui>(this->concreteLeafRanges);
    //auto hConcreteNodeRanges = Span<const Vector2ui>(hConcreteNodeRangesVec);
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

    // Easy part is done
    // We need to actually construct this thing now,
    if constexpr(PG::TransformLogic != PrimTransformType::LOCALLY_CONSTANT_TRANSFORM)
        MultiBuildLBVH(nullptr, hInstanceNodeRangesVec,
                       hConcreteNodeRangesVec, queue);
    else for(const auto& kv : this->workInstances)
    {
        Pair<const CommonKey, const AcceleratorWorkI*> input(kv.first, kv.second.get());
        MultiBuildLBVH(&input, hInstanceNodeRangesVec,
                       hConcreteNodeRangesVec, queue);
    }

    data = DataSoA
    {
        .dCullFace                  = ToConstSpan(dCullFaceFlags),
        .dAlphaMaps                 = ToConstSpan(dAlphaMaps),
        .dLightOrMatKeys            = ToConstSpan(dLightOrMatKeys),
        .dPrimitiveRanges           = ToConstSpan(dPrimitiveRanges),
        .dInstanceTransforms        = ToConstSpan(dTransformKeys),
        .dLeafs                     = ToConstSpan(dLeafs),
        .dNodes                     = ToConstSpan(dNodes),
        .dBoundingBoxes             = ToConstSpan(dNodeAABBs)
    };
    // We have temp memory + async memcopies,
    // we need to wait here.
    queue.Barrier().Wait();
}

template<PrimitiveGroupC PG>
void AcceleratorGroupLBVH<PG>::MultiBuildLBVH(Pair<const CommonKey, const AcceleratorWorkI*>* accelWork,
                                              const std::vector<Vector2ui>& instanceNodeRanges,
                                              const std::vector<Vector2ui>& concreteNodeRanges,
                                              const GPUQueue& queue)
{
    static constexpr bool PER_PRIM_TRANSFORM = TransformLogic == PrimTransformType::PER_PRIMITIVE_TRANSFORM;
    static constexpr uint32_t TPB = StaticThreadPerBlock1D();
    static constexpr uint32_t BLOCK_PER_INSTANCE = 16;

    // First calculate leaf ranges
    std::vector<uint32_t> hLeafSegmentRanges;
    std::vector<uint32_t> hNodeSegmentRanges;
    Span<TransformKey> dLocalTransformKeys;
    if constexpr(PER_PRIM_TRANSFORM)
    {
        CommonKey workIndex = accelWork->first;
        Vector2ui workInstanceRange = Vector2ui(this->workInstanceOffsets[workIndex],
                                                this->workInstanceOffsets[workIndex + 1]);
        dLocalTransformKeys = dTransformKeys.subspan(workInstanceRange[0],
                                                     workInstanceRange[1] - workInstanceRange[0]);

        // Leaf segments
        // For per primitive transform types segment leaf ranges is
        // the instance leaf ranges itself, since we cannot reuse an accelerator
        // for primitives that require "per_primitive_transform"
        auto iLeafRange = this->instanceLeafRanges;
        auto localInstanceLeafRanges = Span<const Vector2ui>(iLeafRange.cbegin() + workInstanceRange[0],
                                                             iLeafRange.cbegin() + workInstanceRange[1]);
        hLeafSegmentRanges.reserve(localInstanceLeafRanges.size() + 1);
        hLeafSegmentRanges.push_back(0);
        for(const Vector2ui& range : localInstanceLeafRanges)
            hLeafSegmentRanges.push_back(range[1]);

        // Node Segments
        // Do the same logic to the nodes as well
        const auto& iNodeRange = instanceNodeRanges;
        auto localInstanceNodeRanges = Span<const Vector2ui>(iNodeRange.cbegin() + workInstanceRange[0],
                                                             iNodeRange.cbegin() + workInstanceRange[1]);

        hNodeSegmentRanges.reserve(localInstanceNodeRanges.size() + 1);
        hNodeSegmentRanges.push_back(0);
        for(const Vector2ui& range : localInstanceNodeRanges)
            hNodeSegmentRanges.push_back(range[1]);
    }
    else
    {
        hLeafSegmentRanges.reserve(this->concreteLeafRanges.size() + 1);
        hLeafSegmentRanges.push_back(0);
        for(const Vector2ui& range : this->concreteLeafRanges)
            hLeafSegmentRanges.push_back(range[1]);

        hNodeSegmentRanges.reserve(concreteNodeRanges.size() + 1);
        hNodeSegmentRanges.push_back(0);
        for(const Vector2ui& range : concreteNodeRanges)
            hNodeSegmentRanges.push_back(range[1]);
    }
    uint32_t processedAccelCount = static_cast<uint32_t>(hLeafSegmentRanges.size() - 1);
    uint32_t totalLeafCount = static_cast<uint32_t>(dAllLeafs.size());

    // Allocate temp memory
    // Construction of LBVH requires sorting (of MortonCodes)
    // calculate that, we also need to reduce the AABBs beforehand,
    // to find and optimal Morton Code delta
    using namespace DeviceAlgorithms;
    using namespace std::string_view_literals;

    // TODO: The memory usage can be optimized
    // by repurposing some buffers but currently no memory issues
    // and this happens in intialization time so fine

    // Specific Part 0
    Span<Byte> dTemp;
    Span<uint32_t> dLeafSegmentRanges;
    Span<uint32_t> dNodeSegmentRanges;
    Span<AABB3> dLeafAABBs;
    Span<AABB3> dAccelAABBs;
    Span<Vector3> dPrimCenters;
    std::array<Span<uint32_t>, 2> dIndices;
    std::array<Span<uint64_t>, 2> dMortonCodes;
    //
    size_t segTRMemSize = SegmentedTransformReduceTMSize<AABB3, PrimitiveKey>(processedAccelCount, queue);
    size_t segSortTMSize = SegmentedRadixSortTMSize<true, uint64_t, uint32_t>(totalLeafCount,
                                                                              processedAccelCount, queue);
    size_t tempMemSize = std::max(segTRMemSize, segSortTMSize);
    // For simplicity we are allocating twice here
    // So we can repurpose
    size_t total = MemAlloc::RequiredAllocation<10>
    ({
        tempMemSize * sizeof(Byte),
        (processedAccelCount + 1) * sizeof(uint32_t),
        (processedAccelCount + 1) * sizeof(uint32_t) ,
        (processedAccelCount) * sizeof(AABB3),
        totalLeafCount * sizeof(Vector3),
        totalLeafCount * sizeof(AABB3),
        totalLeafCount * sizeof(uint64_t),
        totalLeafCount * sizeof(uint64_t),
        totalLeafCount * sizeof(uint32_t),
        totalLeafCount * sizeof(uint32_t)
     });
    DeviceMemory tempMem({queue.Device()}, total, total << 1);
    // TODO: The memory can be further alised thus; reduced in size.
    MemAlloc::AllocateMultiData(std::tie(dTemp, dLeafSegmentRanges,
                                         dNodeSegmentRanges, dAccelAABBs,
                                         dPrimCenters, dLeafAABBs,
                                         dMortonCodes[0], dMortonCodes[1],
                                         dIndices[0], dIndices[1]),
                                tempMem,
                                {tempMemSize,
                                 processedAccelCount + 1,
                                 processedAccelCount + 1,
                                 processedAccelCount,
                                 totalLeafCount, totalLeafCount,
                                 totalLeafCount, totalLeafCount,
                                 totalLeafCount, totalLeafCount});
    // Copy the ranges and lets go!
    queue.MemcpyAsync(dLeafSegmentRanges, Span<const uint32_t>(hLeafSegmentRanges));
    queue.MemcpyAsync(dNodeSegmentRanges, Span<const uint32_t>(hNodeSegmentRanges));

    // This implementation is related to this paper.
    // https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
    //
    // One difference is that all operations are "segmented"
    // meaning a single kernel call will operate on multiple accelerators.
    //
    // In summary:
    // 1. Find prim centers (transformed or not depending on the transform logic)
    //    and AABBs of the accelerator instances
    // 2. Transform reduce to find instance AABBs
    // 3. Calculate temporary morton codes of each primitive using the AABBs + centers
    //    (aka. quantize the AABB to 2^12 segments and find where the prim center is)
    // 4. Do a sort to implicitly construct the LBVH hierarchy (index-id sort)
    // 5. Launch a kernel to explicitly create the node wirings (Figure 4. of the paper)
    // 6. Calculate per-primitive AABBs (again transformed or not depending on the logic)
    // 7. Bottom-up union the AABBs
    //
    // And it should be done.
    // Since we need to do these on base accelerator
    // as well a generic impl should be considered.

    // Now lets start
    // 1. Create Instance AABBs. We will use these
    // to quantize the primitive centers to morton codes.
    // We will also need the AABBs of each prim,
    // do it as well
    // Calculate prim AABBs
    if constexpr(PER_PRIM_TRANSFORM)
    {
        accelWork->second->GeneratePrimitiveAABBs(dLeafAABBs,
                                                  ToConstSpan(dLeafSegmentRanges),
                                                  ToConstSpan(dAllLeafs),
                                                  ToConstSpan(dLocalTransformKeys),
                                                  queue);
    }
    else
    {
        uint32_t blockCount = BLOCK_PER_INSTANCE * processedAccelCount;
        queue.IssueBlockKernel<KCGeneratePrimAABBs<AcceleratorGroupLBVH<PG>>>
        (
            "KCGeneratePrimAABBs"sv,
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = TPB
            },
            // Output
            dLeafAABBs,
            // Inputs
            ToConstSpan(dLeafSegmentRanges),
            Span<const TransformKey>(),
            ToConstSpan(dAllLeafs),
            // Constants
            BLOCK_PER_INSTANCE,
            processedAccelCount,
            typename TransformGroupIdentity::DataSoA{},
            this->pg.SoA()
        );
    }
    // We don't have segmented reduce atm.
    // use segmented transform reduce
    SegmentedTransformReduce(dAccelAABBs, dTemp,
                             ToConstSpan(dLeafAABBs),
                             ToConstSpan(dLeafSegmentRanges),
                             AABB3::Negative(),
                             queue, UnionAABB3Functor(),
                             IdentityFunctor<AABB3>());

    // 2. Calculate the primitive centers
    // (In future maybe surface area's for SAH maybe?)
    if constexpr(PER_PRIM_TRANSFORM)
    {
        accelWork->second->GeneratePrimitiveCenters(dPrimCenters,
                                                    ToConstSpan(dLeafSegmentRanges),
                                                    ToConstSpan(dAllLeafs),
                                                    ToConstSpan(dLocalTransformKeys),
                                                    queue);
    }
    else
    {
        uint32_t blockCount = BLOCK_PER_INSTANCE * processedAccelCount;
        queue.IssueBlockKernel<KCGenPrimCenters<AcceleratorGroupLBVH<PG>>>
        (
            "KCGenPrimCenters"sv,
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = TPB
            },
            // Output
            dPrimCenters,
            // Inputs
            ToConstSpan(dLeafSegmentRanges),
            Span<const TransformKey>(),
            ToConstSpan(dAllLeafs),
            BLOCK_PER_INSTANCE,
            processedAccelCount,
            typename TransformGroupIdentity::DataSoA{},
            this->pg.SoA()

        );
    }
    // 3. Convert these to morton codes
    {
        uint32_t instanceCount = static_cast<uint32_t>(dAccelAABBs.size());
        uint32_t blockCount = instanceCount * BLOCK_PER_INSTANCE;
        queue.IssueBlockKernel<KCGenMortonCode>
        (
            "KCGenMortonCodes"sv,
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = TPB
            },
            // Output
            dMortonCodes[0],
            // Inputs
            ToConstSpan(dLeafSegmentRanges),
            ToConstSpan(dAccelAABBs),
            //
            dPrimCenters,
            BLOCK_PER_INSTANCE
        );
    }
    // 4. Sort these codes to implicitly generate the BVH
    SegmentedIota(dIndices[0], ToConstSpan(dLeafSegmentRanges), 0u, queue);
    uint32_t sortedIndex = SegmentedRadixSort<true, uint64_t, uint32_t>
    (
        dMortonCodes, dIndices, dTemp,
        dLeafSegmentRanges, queue
    );
    if(sortedIndex == 1)
    {
        std::swap(dMortonCodes[0], dMortonCodes[1]);
        std::swap(dIndices[0], dIndices[1]);
    }

    // Alias the memory, indces[1], and mortonCode[1] are not used
    // anymore
    Span<uint32_t> dAtomicCounters = MemAlloc::RepurposeAlloc<uint32_t>(dIndices[1]);
    Span<uint32_t> dLeafParentIndices = MemAlloc::RepurposeAlloc<uint32_t>(dMortonCodes[1]);

    // 5. Now we have a multiple valid morton code lists,
    // Construct the node hierarchy
    {
        uint32_t blockCount = processedAccelCount * BLOCK_PER_INSTANCE;
        queue.IssueBlockKernel<KCConstructLBVHInternalNodes>
        (
            "KCConstructLBVHInternalNodes"sv,
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = TPB
            },
            // Output
            dAllNodes,
            dLeafParentIndices,
            // Inputs
            ToConstSpan(dLeafSegmentRanges),
            ToConstSpan(dNodeSegmentRanges),
            ToConstSpan(dMortonCodes[0]),
            ToConstSpan(dIndices[0]),
            //
            BLOCK_PER_INSTANCE,
            processedAccelCount
        );
    }
    // 6. Finally at AABB union portion now, union the AABBs.
    {
        queue.MemsetAsync(dAtomicCounters, 0x00);
        uint32_t blockCount = BLOCK_PER_INSTANCE * processedAccelCount;
        queue.IssueBlockKernel<KCUnionLBVHBoundingBoxes>
        (
            "KCUnionLBVHBoundingBoxes"sv,
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = TPB
            },
            // Output
            dAllNodeAABBs,
            dAtomicCounters,
            // Inputs
            ToConstSpan(dAllNodes),
            ToConstSpan(dLeafParentIndices),
            ToConstSpan(dLeafSegmentRanges),
            ToConstSpan(dNodeSegmentRanges),
            ToConstSpan(dLeafAABBs),
            //
            BLOCK_PER_INSTANCE,
            processedAccelCount
        );
    }
    // All Done!
    // Wait GPU before dealloc
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
    // TODO: This is wastefull, we do a transform-reduce on leaf level
    // Since we are BVH, root node implictly have the AABB
    // But this code works, but may have a performance bottleneck
    // in future maybe?
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
void AcceleratorGroupLBVH<PG>::CastVisibilityRays(// Output
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