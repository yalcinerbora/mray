#include "AcceleratorLBVH.h"

#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgRadixSort.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgGeneric.h"
#include "Device/GPUAtomic.h"

#include "Core/Error.hpp"

#include "TypeFormat.h"

struct MortonDiffFunctor
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    uint32_t operator()(uint64_t l, uint64_t r) const
    {
        return (l != r) ? 1u : 0u;
    }
};

MRAY_HYBRID MRAY_CGPU_INLINE
LBVHAccelDetail::Delta::Delta(const Span<const uint64_t>& dMCs)
    : dMortonCodes(dMCs)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
int32_t LBVHAccelDetail::Delta::operator()(int32_t i, int32_t j) const
{
    if(j < 0 || j >= static_cast<int32_t>(dMortonCodes.size()))
        return -1;

    uint64_t left = dMortonCodes[i];
    uint64_t right = dMortonCodes[j];

    // Equal morton fallback
    if(left == right)
    {
        left = i;
        right = j;
    }

    uint64_t diffBits = left ^ right;
    return static_cast<int32_t>(Bit::CountLZero(diffBits));
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenMortonCode(// Output
                     MRAY_GRID_CONSTANT const Span<uint64_t> dMortonCodes,
                     // Inputs
                     MRAY_GRID_CONSTANT const Span<const uint32_t> dSegmentRanges,
                     MRAY_GRID_CONSTANT const Span<const AABB3> dInstanceAABBs,
                     //
                     MRAY_GRID_CONSTANT const Span<const Vector3> dAllPrimCenters,
                     // Constants
                     MRAY_GRID_CONSTANT const uint32_t blockPerInstance)
{
    static constexpr auto TPB = StaticThreadPerBlock1D();
    assert(dSegmentRanges.size() == dInstanceAABBs.size() + 1);

    // Block-stride loop
    KernelCallParams kp;
    uint32_t instanceCount = static_cast<uint32_t>(dInstanceAABBs.size());
    uint32_t blockCount = instanceCount * blockPerInstance;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        // Current instance index of this iteration
        uint32_t instanceI = bI / blockPerInstance;
        uint32_t localBI = bI % blockPerInstance;
        //
        uint32_t instanceLocalThreadId = localBI * TPB + kp.threadId;
        uint32_t primPerPass = TPB * blockPerInstance;
        //
        Vector2ui range = Vector2ui(dSegmentRanges[instanceI],
                                    dSegmentRanges[instanceI + 1]);
        auto dLocalPrimCenters = dAllPrimCenters.subspan(range[0],
                                                         range[1] - range[0]);
        auto dLocalMortonCode = dMortonCodes.subspan(range[0],
                                                     range[1] - range[0]);
        // Loop invariant data
        // We will divide the instance local AABB to 12-bit slices
        // hope that the centers do not lay on the same voxel
        Vector3 aabbSize = dInstanceAABBs[instanceI].GeomSpan();
        Float maxSide = aabbSize[aabbSize.Maximum()];
        using namespace Graphics::MortonCode;
        static constexpr uint64_t MaxBits = MaxBits3D<uint64_t>();
        static constexpr Float SliceCount = Float(1ull << MaxBits);
        // TODO: Check if 32-bit float is not enough here (precision)
        Float deltaRecip = SliceCount / maxSide;

        // Finally multi-block primitive loop
        uint32_t totalPrims = static_cast<uint32_t>(dLocalPrimCenters.size());
        for(uint32_t i = instanceLocalThreadId; i < totalPrims;
            i += primPerPass)
        {
            Vector3 center = dLocalPrimCenters[i];
            // Quantize the center
            center = (center  * deltaRecip).Round();
            Vector3ui xyz(center[0], center[1], center[2]);
            dLocalMortonCode[i] = Compose3D<uint64_t>(xyz);
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCConstructLBVHInternalNodes(// Output
                                  MRAY_GRID_CONSTANT const Span<LBVHAccelDetail::LBVHNode> dAllNodes,
                                  // Inputs
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dSegmentRanges,
                                  MRAY_GRID_CONSTANT const Span<const uint64_t> dAllMortonCodes,
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dAllLeafIndices,
                                  // Constants
                                  MRAY_GRID_CONSTANT const uint32_t blockPerInstance,
                                  MRAY_GRID_CONSTANT const uint32_t instanceCount)
{
    using namespace LBVHAccelDetail;
    static constexpr auto TPB = StaticThreadPerBlock1D();
    // Block-stride loop
    KernelCallParams kp;
    uint32_t blockCount = instanceCount * blockPerInstance;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        // Current instance index of this iteration
        uint32_t instanceI = bI / blockPerInstance;
        uint32_t localBI = bI % blockPerInstance;

        if(instanceI >= instanceCount) continue;

        //
        int32_t instanceLocalThreadId = static_cast<int32_t>(localBI * TPB + kp.threadId);
        int32_t primPerPass = static_cast<int32_t>(TPB * blockPerInstance);
        //
        Vector2ui range = Vector2ui(dSegmentRanges[instanceI],
                                    dSegmentRanges[instanceI + 1]);
        auto dLocalNodes = dAllNodes.subspan(range[0],
                                             range[1] - range[0]);
        auto dLocalCodes = dAllMortonCodes.subspan(range[0],
                                                   range[1] - range[0]);
        auto dLocalIndices = dAllLeafIndices.subspan(range[0],
                                                     range[1] - range[0]);

        // Delta function that is described in the paper
        Delta delta(dLocalCodes);

        int32_t totalPrims = static_cast<int32_t>(dLocalNodes.size());
        // Edge case: Single primitive
        if(totalPrims == 1 && instanceLocalThreadId == 0)
        {
            LBVHNode& myNode = dLocalNodes[0];
            myNode.leftIndex = ChildIndex::CombinedKey(IS_LEAF, 0);
            myNode.rightIndex = ChildIndex::InvalidKey();
            myNode.parentIndex = std::numeric_limits<uint32_t>::max();
        }
        // Edge case: Two primitives? (Paper says [0, n-2] where n is leaf count)
        else if(totalPrims == 2 && instanceLocalThreadId == 0)
        {
            LBVHNode& myNode = dLocalNodes[0];
            myNode.leftIndex = ChildIndex::CombinedKey(IS_LEAF, 0);
            myNode.rightIndex = ChildIndex::CombinedKey(IS_LEAF, 1);
            myNode.parentIndex = std::numeric_limits<uint32_t>::max();
        }
        // Common case
        else for(int32_t i = instanceLocalThreadId; i < totalPrims; i += primPerPass)
        {
            // From paper's "Figure 4"
            int32_t d = ((delta(i, i + 1) - delta(i, i - 1)) < 0) ? 0 : -1;

            // Compute upper bound
            int32_t deltaMin = delta(i, i - d);
            int32_t lMax = 2;
            while(delta(i, i + lMax * d) > deltaMin)
                lMax <<= 1;

            // Binary search to find end
            int32_t length = 0;
            for(int32_t t = lMax >> 1; t != 0; t >>= 1)
            {
                if(delta(i, i + (length + t) * d) > deltaMin)
                    length += t;
            }
            int32_t j = i + length * d;

            // Binary search to find split
            int32_t deltaRange = delta(i, j);
            int32_t splitOffset = 0;
            for(int32_t t = Math::DivideUp(length, 2);
                t != 0; t = Math::DivideUp(t, 2))
            {
                if(delta(i, i + (splitOffset + t) * d) > deltaRange)
                    splitOffset += t;
            }
            int32_t gamma = i + splitOffset * d + std::min(d, 0);

            if(gamma < 0 || gamma >= totalPrims)
            {
                printf("WTF\n");
            }

            // Finally write
            LBVHNode& myNode = dLocalNodes[i];
            if(std::min(i, j) == gamma)
                myNode.leftIndex = ChildIndex::CombinedKey(IS_LEAF, dLocalIndices[gamma]);
            else
            {
                myNode.leftIndex = ChildIndex::CombinedKey(IS_INTERNAL, gamma);
                dLocalNodes[gamma].parentIndex = i;
            }
            //
            if(std::max(i, j) == (gamma + 1))
                myNode.rightIndex = ChildIndex::CombinedKey(IS_LEAF, dLocalIndices[gamma + 1]);
            else
            {
                myNode.rightIndex = ChildIndex::CombinedKey(IS_INTERNAL, gamma + 1);
                dLocalNodes[gamma + 1].parentIndex = i;
            }

            // Edge case: if root node, set parent to "null"
            if(i == 0) myNode.parentIndex = std::numeric_limits<uint32_t>::max();
            // All Done!
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCUnionLBVHBoundingBoxes(// I-O
                              MRAY_GRID_CONSTANT const Span<LBVHAccelDetail::LBVHBoundingBox> dAllNodeAABBs,
                              MRAY_GRID_CONSTANT const Span<LBVHAccelDetail::LBVHNode> dAllNodes,
                              MRAY_GRID_CONSTANT const Span<uint32_t> dAtomicCounters,
                              // Inputs
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dSegmentRanges,
                              MRAY_GRID_CONSTANT const Span<const AABB3> dAllLeafAABBs,
                              // Constants
                              MRAY_GRID_CONSTANT const uint32_t blockPerInstance,
                              MRAY_GRID_CONSTANT const uint32_t instanceCount)
{
    using namespace LBVHAccelDetail;
    static constexpr auto TPB = StaticThreadPerBlock1D();
    // Block-stride loop
    KernelCallParams kp;
    uint32_t blockCount = instanceCount * blockPerInstance;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        // Current instance index of this iteration
        uint32_t instanceI = bI / blockPerInstance;
        uint32_t localBI = bI % blockPerInstance;

        if(instanceI >= instanceCount) continue;

        //
        int32_t instanceLocalThreadId = static_cast<int32_t>(localBI * TPB + kp.threadId);
        int32_t primPerPass = static_cast<int32_t>(TPB * blockPerInstance);
        //
        Vector2ui range = Vector2ui(dSegmentRanges[instanceI],
                                    dSegmentRanges[instanceI + 1]);
        auto dLocalNodes = dAllNodes.subspan(range[0],
                                             range[1] - range[0]);
        auto dLocalAABBs = dAllLeafAABBs.subspan(range[0],
                                                 range[1] - range[0]);
        auto dLocalCounters = dAtomicCounters.subspan(range[0],
                                                      range[1] - range[0]);
        // Node AABBs are volatile since the writes needs to be coherent
        // between threads. This + thread fence hopefully should suffice
        // the parallel aabb union.
        //
        // volatile should make the data to punch-through the cache hierarchy and,
        // threadfence should prevent read/write reordering.
        // If not we can fall back to atomic
        using AABBVolatileSpan = Span<volatile LBVHAccelDetail::LBVHBoundingBox>;
        auto dLocalNodeAABBs = AABBVolatileSpan(dAllNodeAABBs.data() + range[0],
                                                size_t(range[1] - range[0]));

        auto FetchAABB = [&](ChildIndex cIndex) -> AABB3
        {
            bool isLeaf = (cIndex.FetchBatchPortion() == IS_LEAF);
            uint32_t index = cIndex.FetchIndexPortion();
            // Due to volatile, we need to load like this
            // (without adding volatile overloads to functions)
            Float min0 = dLocalNodeAABBs[index].min[0];
            Float min1 = dLocalNodeAABBs[index].min[1];
            Float min2 = dLocalNodeAABBs[index].min[2];
            //
            Float max0 = dLocalNodeAABBs[index].min[0];
            Float max1 = dLocalNodeAABBs[index].min[1];
            Float max2 = dLocalNodeAABBs[index].min[2];
            //
            AABB3 aabb = (isLeaf) ? dLocalAABBs[index]
                                  : AABB3(Vector3(min0, min1, min2),
                                          Vector3(max0, max1, max2));
            return aabb;
        };

        // Actual loop
        int32_t totalPrims = static_cast<int32_t>(dLocalNodes.size());
        for(int32_t i = instanceLocalThreadId; i < totalPrims;
            i += primPerPass)
        {
            //
            uint32_t nodeIndex = i;
            while(nodeIndex != std::numeric_limits<uint32_t>::max())
            {
                const LBVHNode& node = dLocalNodes[nodeIndex];
                volatile LBVHBoundingBox& bbox = dLocalNodeAABBs[nodeIndex];
                uint32_t result = DeviceAtomic::AtomicAdd(dLocalCounters[i], 1u);
                // Last one come to the party, cleanup after the other guys
                if(result == 1)
                {
                    AABB3 lAABB = FetchAABB(node.leftIndex);
                    AABB3 rAABB = FetchAABB(node.rightIndex);
                    AABB3 aabb = lAABB.Union(rAABB);

                    // Using volatile to ensure visibility
                    // (+ thread fence)
                    bbox.min[0] = aabb.Min()[0];
                    bbox.min[1] = aabb.Min()[1];
                    bbox.min[2] = aabb.Min()[2];
                    bbox.max[0] = aabb.Max()[0];
                    bbox.max[1] = aabb.Max()[1];
                    bbox.max[2] = aabb.Max()[2];
                    // We need to prevent mem reordering
                    ThreadFenceGrid();
                    // Go up
                    nodeIndex = node.parentIndex;
                }
                // Kill the thread
                else break;
            }
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCIntersectBaseLBVH(// Output
                         MRAY_GRID_CONSTANT const Span<CommonKey> dAccelKeys,
                         // I-O
                         MRAY_GRID_CONSTANT const Span<uint32_t> dBitStackStates,
                         MRAY_GRID_CONSTANT const Span<uint32_t> dPrevNodeIndices,
                         // Input
                         MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                         MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                         // Constants
                         MRAY_GRID_CONSTANT const Span<const AcceleratorKey> dLeafKeys,
                         MRAY_GRID_CONSTANT const Span<const AABB3> dLeafAABBs,
                         MRAY_GRID_CONSTANT const Span<const LBVHAccelDetail::LBVHNode> dNodes)
{
    //using Bit::FetchSubPortion;
    //using AU32 = std::array<uint32_t, 2>;
    //static constexpr AU32 StackStateRange = {0u, BaseAcceleratorLBVH::StackBitCount};
    //static constexpr AU32 DepthRange = {BaseAcceleratorLBVH::StackBitCount, 32u};

    //assert(dLeafAABBs.size() == dLeafKeys.size());
    //KernelCallParams kp;
    //uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    //for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    //{
    //    RayIndex index = dRayIndices[i];
    //    auto [ray, tMM] = RayFromGMem(dRays, index);

    //    AcceleratorKey foundKey = AcceleratorKey::InvalidKey();
    //    uint32_t stackState = dBitStackStates[index];
    //    uint32_t currentNodeIndex = dPrevNodeIndices[index];
    //    using namespace LBVHAccelDetail;
    //    BitStack bitStack(FetchSubPortion(stackState, StackStateRange),
    //                      FetchSubPortion(stackState, DepthRange));
    //    // Traversal
    //    uint32_t newPrevIndex = TraverseLBVH(bitStack, tMM, ray, dNodes, true,
    //                                         currentNodeIndex,
    //    [&](Vector2& tMM, uint32_t leafIndex)
    //    {
    //        AABB3 aabb = dLeafAABBs[leafIndex];
    //        bool doIntersect = ray.IntersectsAABB(aabb.Min(), aabb.Max(), tMM);
    //        if(doIntersect)
    //            foundKey = dLeafKeys[leafIndex];
    //        // Stop traversal delegate to the inner accelerator
    //        return doIntersect;
    //    });

    //    // Save the traversal stack state
    //    static constexpr auto StackBits = BaseAcceleratorLBVH::StackBitCount;
    //    static constexpr auto DepthBits = BaseAcceleratorLBVH::DepthBitCount;
    //    dBitStackStates[index] = bitStack.CompressState<StackBits, DepthBits>();
    //    dPrevNodeIndices[index] = newPrevIndex;
    //    // Return the currently found key
    //    dAccelKeys[index] = static_cast<CommonKey>(foundKey);
    //}
}

std::string_view BaseAcceleratorLBVH::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "BVH"sv;
    return BaseAccelTypeName<Name>;
}

AABB3 BaseAcceleratorLBVH::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    assert(instanceOffsets.size() == (generatedAccels.size() + 1));

    // Allocate persistent memory
    size_t instanceCount = instanceOffsets.back();
    // Assume a complete tree, and allocate conservatively
    size_t conservativeNodeCount = Math::NextPowerOfTwo(instanceCount) << 1u;
    conservativeNodeCount -= 1;

    MemAlloc::AllocateMultiData(std::tie(dLeafKeys, dLeafAABBs, dNodes),
                                accelMem,
                                {instanceCount, instanceCount,
                                conservativeNodeCount});

    // Write leafs and transformed aabbs to the array
    size_t i = 0;
    GPUQueueIteratorRoundRobin qIt(gpuSystem);
    for(const auto& accGroup : generatedAccels)
    {
        AcceleratorGroupI* aGroup = accGroup.second.get();
        size_t localCount = instanceOffsets[i + 1] - instanceOffsets[i];
        auto dAABBRegion = dLeafAABBs.subspan(instanceOffsets[i],
                                              localCount);
        auto dLeafRegion = dLeafKeys.subspan(instanceOffsets[i], localCount);
        aGroup->WriteInstanceKeysAndAABBs(dAABBRegion, dLeafRegion, qIt.Queue());
        i++;
        qIt.Next();
    }

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    //sceneAABB;// = LBVHAccelDetail::MulitBuildLBVH(queue)[0];

    return sceneAABB;
}

void BaseAcceleratorLBVH::AllocateForTraversal(size_t maxRayCount)
{
    maxPartitionCount = std::transform_reduce(generatedAccels.cbegin(),
                                              generatedAccels.cend(),
                                              size_t(0), std::plus{},
                                              [](const auto& pair)
    {
        return pair.second->InstanceTypeCount();
    });

    rayPartitioner = RayPartitioner(gpuSystem, static_cast<uint32_t>(maxRayCount),
                                    static_cast<uint32_t>(maxPartitionCount));
    MemAlloc::AllocateMultiData(std::tie(dTraversalStack), stackMem, {maxRayCount});
}

void BaseAcceleratorLBVH::CastRays(// Output
                                   Span<HitKeyPack> dHitIds,
                                   Span<MetaHit> dHitParams,
                                   // I-O
                                   Span<BackupRNGState> rngStates,
                                   Span<RayGMem> dRays,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    using namespace std::string_view_literals;
    queue.MemsetAsync(dTraversalStack, 0x00);

    // Initialize the ray partitioner
    uint32_t currentRayCount = static_cast<uint32_t>(dRays.size());
    uint32_t partitionCount = static_cast<uint32_t>(maxPartitionCount);
    auto [dCurrentIndices, dCurrentKeys] = rayPartitioner.Start(currentRayCount, partitionCount);
    // Copy the ray indices to the local buffer, normally we could utilize
    // global ray partitioner (if available) but
    // - Not all renderers (very simple ones probably) may not have a partitioner
    // - OptiX (or equavilent on other hardwares hopefully in the future) already
    //   does two-level acceleration in hardware, so we dont need to do this
    queue.MemcpyAsync(dCurrentIndices, dRayIndices);
    // Continiously do traverse/partition untill all rays are missed
    while(currentRayCount != 0)
    {
        //queue.IssueSaturatingKernel<KCIntersectBaseLBVH>
        //(
        //    "(A)LBVHRayCast"sv,
        //    KernelIssueParams{.workCount = static_cast<uint32_t>(dRayIndices.size())},
        //    // Output
        //    dCurrentKeys,
        //    // I-O
        //    dTraversalStack,
        //    // Input
        //    dRays,
        //    dCurrentIndices,
        //    // Constants
        //    ToConstSpan(dLeafs),
        //    ToConstSpan(dAABBs)
        //);

        static constexpr CommonKey IdBits = AcceleratorKey::IdBits;
        static constexpr CommonKey BatchBits = AcceleratorKey::IdBits;
        auto batchRange = Vector2ui(BatchBits, BatchBits + maxBitsUsedOnKey[0]);
        auto idRange = Vector2ui(IdBits, IdBits + maxBitsUsedOnKey[1]);
        auto
        [
            hPartitionCount,
            //
            isHostVisible,
            hPartitionOffsets,
            hKeys,
            //
            dIndices,
            dKeys
        ] = rayPartitioner.MultiPartition(dCurrentKeys,
                                          dCurrentIndices,
                                          idRange,
                                          batchRange,
                                          queue, false);
        assert(isHostVisible == true);
        queue.Barrier().Wait();
        for(uint32_t pIndex = 0; pIndex < hPartitionCount[0]; pIndex++)
        {
            AcceleratorKey key(hKeys[pIndex]);
            // This means we could not find the next second-level
            // acceleration structure, meaning these rays are are finished traversing
            if(key == AcceleratorKey::InvalidKey())
            {
                // This should be the last item due to invalid key being INT_MAX
                assert(pIndex == hPartitionCount[0] - 1);
                currentRayCount = hPartitionOffsets[pIndex];
                dCurrentKeys = dKeys.subspan(0, currentRayCount);
                dCurrentIndices = dIndices.subspan(0, currentRayCount);
            }
            else
            {
                // Normal work, find the accel group issue the kernel
                uint32_t partitionStart = hPartitionOffsets[pIndex];
                uint32_t localSize = hPartitionOffsets[pIndex + 1] - partitionStart;

                //....
                Span<const RayIndex> dLocalIndices = ToConstSpan(dCurrentIndices.subspan(partitionStart,
                                                                                         localSize));
                Span<const CommonKey> dLocalKeys = ToConstSpan(dCurrentKeys.subspan(partitionStart,
                                                                                    localSize));
                auto accelGroupOpt = accelInstances.at(key.FetchBatchPortion());
                if(!accelGroupOpt)
                {
                    throw MRayError("BaseAccelerator: Unknown accelerator key {}", HexKeyT(key));
                }
                AcceleratorGroupI* accelGroup = accelGroupOpt.value().get();
                accelGroup->CastLocalRays(// Output
                                          dHitIds,
                                          dHitParams,
                                          // I-O
                                          rngStates,
                                          //
                                          dRays,
                                          dLocalIndices,
                                          dLocalKeys,
                                          //
                                          key.FetchBatchPortion(),
                                          queue);
            }
        }
    }
}

void BaseAcceleratorLBVH::CastShadowRays(// Output
                                         Bitspan<uint32_t> dIsVisibleBuffer,
                                         Bitspan<uint32_t> dFoundMediumInterface,
                                         // I-O
                                         Span<BackupRNGState> rngStates,
                                         // Input
                                         Span<const RayIndex> dRayIndices,
                                         Span<const RayGMem> dShadowRays,
                                         const GPUQueue& queue)
{

}

void BaseAcceleratorLBVH::CastLocalRays(// Output
                                        Span<HitKeyPack> dHitIds,
                                        Span<MetaHit> dHitParams,
                                        // I-O
                                        Span<BackupRNGState> rngStates,
                                        // Input
                                        Span<const RayGMem> dRays,
                                        Span<const RayIndex> dRayIndices,
                                        Span<const AcceleratorKey> dAccelIdPacks,
                                        const GPUQueue& queue)
{

}

size_t BaseAcceleratorLBVH::GPUMemoryUsage() const
{
    size_t totalSize = accelMem.Size() + stackMem.Size();
    for(const auto& [_, accelGroup] : this->generatedAccels)
    {
        totalSize += accelGroup->GPUMemoryUsage();
    }
    return totalSize;
}