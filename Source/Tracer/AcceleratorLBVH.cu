#include "AcceleratorLBVH.h"

#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgRadixSort.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgGeneric.h"
#include "Device/GPUAtomic.h"

#include "TypeFormat.h"

// Explicitly instantiate these
// These are used by "AcceleratorGroupLBVH::MultiBuildBVH"
template size_t
DeviceAlgorithms::SegmentedTransformReduceTMSize<AABB3, PrimitiveKey>(size_t, const GPUQueue&);

template size_t
DeviceAlgorithms::SegmentedRadixSortTMSize<true, uint64_t, uint32_t>(size_t, size_t, const GPUQueue&);

template void
DeviceAlgorithms::
SegmentedTransformReduce
<AABB3, AABB3, UnionAABB3Functor, IdentityFunctor<AABB3>>
(
    Span<AABB3>,
    Span<Byte>,
    Span<const AABB3>,
    Span<const uint32_t>,
    const AABB3&,
    const GPUQueue&,
    UnionAABB3Functor&& ,
    IdentityFunctor<AABB3>&&
);

template uint32_t
DeviceAlgorithms::SegmentedRadixSort<true, uint64_t, uint32_t>
(
    Span<Span<uint64_t>, 2>,
    Span<Span<uint32_t>, 2>,
    Span<Byte>,
    Span<const uint32_t>,
    const GPUQueue&,
    const Vector2ui&
);

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

    assert(i >= 0);
    assert(j >= 0);
    uint64_t left = dMortonCodes[uint32_t(i)];
    uint64_t right = dMortonCodes[uint32_t(j)];

    // Equal morton fallback
    if(left == right)
    {
        left = uint64_t(i);
        right = uint64_t(j);
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

        if(instanceI >= instanceCount) continue;

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
        AABB3 aabb = dInstanceAABBs[instanceI];
        Vector3 aabbSize = aabb.GeomSpan();
        Float maxSide = aabbSize[aabbSize.Maximum()];
        using namespace Graphics::MortonCode;
        static constexpr uint32_t MaxBits = uint32_t(MaxBits3D<uint64_t>());
        static constexpr uint32_t LastValue = (1u << MaxBits) - 1;
        // We do double calculation here since we need to divide
        // with 2^20 (3D 64-bit morton code fits 20-bit integers)
        // This will be slow on the GPU but I've tried to
        // minimize it as much as possible...
        static constexpr double SliceCount = double(1ull << MaxBits);
        // TODO: Check if 32-bit float is not enough here (precision)
        double deltaRecip = SliceCount / static_cast<double>(maxSide);
        Vector3 bottomLeft = aabb.Min();

        // Finally multi-block primitive loop
        uint32_t totalPrims = static_cast<uint32_t>(dLocalPrimCenters.size());
        for(uint32_t i = instanceLocalThreadId; i < totalPrims;
            i += primPerPass)
        {
            Vector3 center = dLocalPrimCenters[i];
            Vector3 diff = center - bottomLeft;
            // Diff still can have numeric errors
            // Make it positive
            diff = Vector3::Max(diff, Vector3::Zero());
            // Quantize the center relative to the AABB
            auto result = Vector3(Vector3d(diff) * deltaRecip).Round();
            Vector3ui xyz(result[0], result[1], result[2]);
            xyz = xyz.Clamp(0u, LastValue);
            uint64_t code = Compose3D<uint64_t>(xyz);
            dLocalMortonCode[i] = code;
        }
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCConstructLBVHInternalNodes(// Output
                                  MRAY_GRID_CONSTANT const Span<LBVHAccelDetail::LBVHNode> dAllNodes,
                                  MRAY_GRID_CONSTANT const Span<uint32_t> dAllLeafParentIndices,
                                  // Inputs
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dLeafSegmentRanges,
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dNodeSegmentRanges,
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
        Vector2ui leafRange = Vector2ui(dLeafSegmentRanges[instanceI],
                                        dLeafSegmentRanges[instanceI + 1]);
        Vector2ui nodeRange = Vector2ui(dNodeSegmentRanges[instanceI],
                                        dNodeSegmentRanges[instanceI + 1]);
        auto dLocalNodes = dAllNodes.subspan(nodeRange[0],
                                             nodeRange[1] - nodeRange[0]);
        auto dLocalLeafParentIndices = dAllLeafParentIndices.subspan(leafRange[0],
                                                                     leafRange[1] - leafRange[0]);
        auto dLocalCodes = dAllMortonCodes.subspan(leafRange[0],
                                                   leafRange[1] - leafRange[0]);
        auto dLocalIndices = dAllLeafIndices.subspan(leafRange[0],
                                                     leafRange[1] - leafRange[0]);


        // Delta function that is described in the paper
        Delta delta(dLocalCodes);

        int32_t totalLeafs = static_cast<int32_t>(dLocalCodes.size());
        int32_t totalNodes = static_cast<int32_t>(dLocalNodes.size());
        // Edge case: Single primitive
        if(totalLeafs == 1 && instanceLocalThreadId == 0)
        {
            LBVHNode& myNode = dLocalNodes[0];
            myNode.leftIndex = ChildIndex::CombinedKey(IS_LEAF, 0);
            myNode.rightIndex = ChildIndex::InvalidKey();
            myNode.parentIndex = std::numeric_limits<uint32_t>::max();
        }
        // Common case
        else for(int32_t i = instanceLocalThreadId; i < totalNodes; i += primPerPass)
        {
            // From paper's "Figure 4"
            int32_t diff = delta(i, i + 1) - delta(i, i - 1);
            int32_t d = (diff < 0) ? -1 : 1;

            // Compute upper bound
            int32_t deltaMin = delta(i, i - d);
            int32_t lMax = 2;
            while(delta(i, i + lMax * d) > deltaMin)
                lMax <<= 1;

            // Binary search to find end
            int32_t l = 0;
            for(int32_t t = lMax >> 1; t != 0; t >>= 1)
            {
                if(delta(i, i + (l + t) * d) > deltaMin)
                    l += t;
            }
            int32_t j = i + l * d;

            // Binary search to find split
            // Our divide up never reaches to zero
            // so adding a small function here
            auto AdvanceCeil = [](int32_t i)
            {
                if(i == 1) return 0;
                return Math::DivideUp(i, 2);
            };
            int32_t s = 0;
            int32_t deltaNode = delta(i, j);
            for(int32_t t = Math::DivideUp(l, 2);
                t != 0; t = AdvanceCeil(t))
            {
                if(delta(i, i + (s + t) * d) > deltaNode)
                    s += t;
            }
            int32_t gamma = i + s * d + std::min(d, 0);
            assert(gamma >= 0 && gamma < totalNodes);

            // Finally write
            LBVHNode& myNode = dLocalNodes[uint32_t(i)];
            if(std::min(i, j) == gamma)
            {
                uint32_t indirectLeafIndex = dLocalIndices[uint32_t(gamma)];
                myNode.leftIndex = ChildIndex::CombinedKey(IS_LEAF, indirectLeafIndex);
                dLocalLeafParentIndices[indirectLeafIndex] = uint32_t(i);
            }
            else
            {
                myNode.leftIndex = ChildIndex::CombinedKey(IS_INTERNAL, uint32_t(gamma));
                dLocalNodes[uint32_t(gamma)].parentIndex = uint32_t(i);
            }
            //
            if(std::max(i, j) == (gamma + 1))
            {
                uint32_t indirectLeafIndex = dLocalIndices[uint32_t(gamma + 1)];
                myNode.rightIndex = ChildIndex::CombinedKey(IS_LEAF, indirectLeafIndex);
                dLocalLeafParentIndices[indirectLeafIndex] = uint32_t(i);
            }
            else
            {
                myNode.rightIndex = ChildIndex::CombinedKey(IS_INTERNAL, uint32_t(gamma + 1));
                dLocalNodes[uint32_t(gamma + 1)].parentIndex = uint32_t(i);
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
                              MRAY_GRID_CONSTANT const Span<uint32_t> dAtomicCounters,
                              // Inputs
                              MRAY_GRID_CONSTANT const Span<const LBVHAccelDetail::LBVHNode> dAllNodes,
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dAllLeafParentIndices,
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dLeafSegmentRanges,
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dNodeSegmentRanges,
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
        Vector2ui leafRange = Vector2ui(dLeafSegmentRanges[instanceI],
                                        dLeafSegmentRanges[instanceI + 1]);
        Vector2ui nodeRange = Vector2ui(dNodeSegmentRanges[instanceI],
                                        dNodeSegmentRanges[instanceI + 1]);
        auto dLocalNodes = dAllNodes.subspan(nodeRange[0],
                                             nodeRange[1] - nodeRange[0]);
        auto dLocalCounters = dAtomicCounters.subspan(nodeRange[0],
                                                      nodeRange[1] - nodeRange[0]);
        auto dLocalLeafAABBs = dAllLeafAABBs.subspan(leafRange[0],
                                                     leafRange[1] - leafRange[0]);
        auto dLocalLeafParentIndices = dAllLeafParentIndices.subspan(leafRange[0],
                                                                     leafRange[1] - leafRange[0]);
        // Node AABBs are volatile since the writes needs to be coherent
        // between threads. This + thread fence hopefully should suffice
        // the parallel aabb union.
        //
        // volatile should make the data to punch-through the cache hierarchy and,
        // threadfence should prevent read/write reordering.
        // If not we can fall back to atomic
        //
        // std::span<volatile T> where T is a compound type (class?)
        // is ill-formed.
        //
        // NVCC did not care but MSVC did care. Falling back to C
        auto GetLocalAABBVolatile = [=](uint32_t i) -> volatile LBVHAccelDetail::LBVHBoundingBox&
        {
            using AABBVolatileSpanPtr = volatile LBVHAccelDetail::LBVHBoundingBox*;
            auto dLocalNodeAABBs = AABBVolatileSpanPtr(dAllNodeAABBs.data() + nodeRange[0]);
            assert(i < nodeRange[1] - nodeRange[0]);
            return dLocalNodeAABBs[i];
        };

        auto FetchAABB = [&](ChildIndex cIndex) -> AABB3
        {
            bool isLeaf = (cIndex.FetchBatchPortion() == IS_LEAF);
            uint32_t index = cIndex.FetchIndexPortion();
            if(!isLeaf)
            {
                // Due to volatile, we need to load like this
                // (without adding volatile overloads to functions)
                Float min0 = GetLocalAABBVolatile(index).min[0];
                Float min1 = GetLocalAABBVolatile(index).min[1];
                Float min2 = GetLocalAABBVolatile(index).min[2];
                //
                Float max0 = GetLocalAABBVolatile(index).max[0];
                Float max1 = GetLocalAABBVolatile(index).max[1];
                Float max2 = GetLocalAABBVolatile(index).max[2];
                return AABB3(Vector3(min0, min1, min2),
                             Vector3(max0, max1, max2));
            }
            else return dLocalLeafAABBs[index];
        };

        // Edge case: If there is a single primitive, we need to set the counter to one
        // Paper does not discuss this, for single leaf node, we create a single intermediate node
        // to make the tracing code simpler (use do a stackless traversal so it is complex already)
        int32_t totalLeafs = static_cast<int32_t>(dLocalLeafAABBs.size());
        if(totalLeafs == 1 && instanceLocalThreadId < 1)
        {
            volatile LBVHBoundingBox& bbox = GetLocalAABBVolatile(0);
            AABB3 aabb = dLocalLeafAABBs[0];
            bbox.min[0] = aabb.Min()[0];
            bbox.min[1] = aabb.Min()[1];
            bbox.min[2] = aabb.Min()[2];
            bbox.max[0] = aabb.Max()[0];
            bbox.max[1] = aabb.Max()[1];
            bbox.max[2] = aabb.Max()[2];
        }
        // Normal Case
        else for(int32_t i = instanceLocalThreadId; i < totalLeafs;
                 i += primPerPass)
        {
            uint32_t nodeIndex = dLocalLeafParentIndices[uint32_t(i)];
            while(nodeIndex != std::numeric_limits<uint32_t>::max())
            {
                const LBVHNode& node = dLocalNodes[nodeIndex];
                volatile LBVHBoundingBox& bbox = GetLocalAABBVolatile(nodeIndex);
                uint32_t result = DeviceAtomic::AtomicAdd(dLocalCounters[nodeIndex], 1u);
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
                         MRAY_GRID_CONSTANT const Span<const LBVHAccelDetail::LBVHNode> dNodes,
                         MRAY_GRID_CONSTANT const Span<const LBVHAccelDetail::LBVHBoundingBox> dBoxes)
{
    using Bit::FetchSubPortion;
    using AU32 = std::array<uint32_t, 2>;
    static constexpr AU32 StackStateRange = {0u, BaseAcceleratorLBVH::StackBitCount};
    static constexpr AU32 DepthRange = {BaseAcceleratorLBVH::StackBitCount, 32u};

    assert(dLeafAABBs.size() == dLeafKeys.size());
    KernelCallParams kp;
    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayIndex index = dRayIndices[i];
        auto [ray, initialTMM] = RayFromGMem(dRays, index);

        AcceleratorKey foundKey = AcceleratorKey::InvalidKey();
        uint32_t stackState = dBitStackStates[index];
        uint32_t currentNodeIndex = dPrevNodeIndices[index];
        using namespace LBVHAccelDetail;
        BitStack bitStack(FetchSubPortion(stackState, StackStateRange),
                          FetchSubPortion(stackState, DepthRange));
        // Traversal
        uint32_t newPrevIndex = TraverseLBVH<BaseAcceleratorLBVH::StackBitCount>
        (
            bitStack, dNodes, dBoxes,
            initialTMM, ray, currentNodeIndex,
            [&](Vector2& tMM, uint32_t leafIndex)
            {
                AABB3 aabb = dLeafAABBs[leafIndex];
                bool doIntersect = ray.IntersectsAABB(aabb.Min(), aabb.Max(), tMM);
                if(doIntersect)
                    foundKey = dLeafKeys[leafIndex];
                // Stop traversal delegate to the inner accelerator
                return doIntersect;
            }
        );

        // Save the traversal stack state
        static constexpr auto StackBits = BaseAcceleratorLBVH::StackBitCount;
        static constexpr auto DepthBits = BaseAcceleratorLBVH::DepthBitCount;
        dBitStackStates[index] = bitStack.CompressState<StackBits, DepthBits>();
        dPrevNodeIndices[index] = newPrevIndex;
        // Return the currently found key
        dAccelKeys[i] = static_cast<CommonKey>(foundKey);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenAABBCenters(// Outputs
                      Span<Vector3> dAABBCenters,
                      // Inputs
                      const Span<const AABB3> dLeafAABBs)
{
    assert(dAABBCenters.size() == dLeafAABBs.size());
   KernelCallParams kp;
   uint32_t instanceCount = uint32_t(dLeafAABBs.size());
   for(uint32_t i = kp.GlobalId(); i < instanceCount; i += kp.TotalSize())
   {
       dAABBCenters[i] = dLeafAABBs[i].Centroid();
   }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCInitializeBitStack(Span<uint32_t> dBitStacksStates)
{
    LBVHAccelDetail::BitStack stack(0, BaseAcceleratorLBVH::StackBitCount);
    //
    KernelCallParams kp;
    uint32_t total = uint32_t(dBitStacksStates.size());
    for(uint32_t i = kp.GlobalId(); i < total; i += kp.TotalSize())
    {
        dBitStacksStates[i] = stack.CompressState
        <
            BaseAcceleratorLBVH::StackBitCount,
            BaseAcceleratorLBVH::DepthBitCount
        >();
    }
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
    // By definition LBVH has n-1 nodes
    size_t nodeCount = std::max(instanceCount - 1u, size_t(1));

    MemAlloc::AllocateMultiData(std::tie(dLeafKeys, dLeafAABBs,
                                         dNodes, dBoundingBoxes),
                                accelMem,
                                {instanceCount, instanceCount,
                                nodeCount, nodeCount});

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
    // Wait all queues
    gpuSystem.SyncAll();

    // Lets start, the algorithm chain is defined in the AcceleratorGroup,
    // refer to it for clarification.
    using namespace DeviceAlgorithms;
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    // One difference is that entire algorithm is segmented
    // we have a single instance accelerator, so we have a single instance.
    // There may be slight performance loss due to it but this is
    // initialization-time operation so its fine I guess?
    //
    // As a ad-hoc measure, we define a large "blockPerSegment" to
    // fully saturate the GPU.
    uint32_t blockPerSegment = queue.Device()->SMCount() * 16u;

    size_t reduceTMSize = ReduceTMSize<AABB3>(instanceCount, queue);
    size_t sortTMSize = RadixSortTMSize<true, uint64_t, uint32_t>(instanceCount, queue);
    size_t tmSize = std::max(reduceTMSize, sortTMSize);

    // Temp memory
    // TODO: Too much memory, some memory can be ailased
    // since we use a single queue
    Span<AABB3> dSceneAABB;
    Span<Byte> dTemp;
    Span<uint32_t> dLeafSegmentRange;
    Span<uint32_t> dNodeSegmentRange;
    Span<Vector3> dAABBCenters;
    std::array<Span<uint32_t>, 2> dIndices;
    std::array<Span<uint64_t>, 2> dMortonCodes;

    size_t total = MemAlloc::RequiredAllocation<10>
    ({
        tmSize * sizeof(Byte),
        1 * sizeof(AABB3),
        2 * sizeof(uint32_t) ,
        2 * sizeof(uint32_t),
        instanceCount * sizeof(Vector3),
        instanceCount * sizeof(uint64_t),
        instanceCount * sizeof(uint64_t),
        instanceCount * sizeof(uint32_t),
        instanceCount * sizeof(uint32_t)
        });
    DeviceMemory tempMem({queue.Device()}, total, total << 1);
    MemAlloc::AllocateMultiData(std::tie(dTemp, dSceneAABB,
                                         dLeafSegmentRange, dNodeSegmentRange,
                                         dAABBCenters,
                                         dIndices[0], dIndices[1],
                                         dMortonCodes[0], dMortonCodes[1]),
                                tempMem,
                                {tmSize, 1, 2, 2,
                                instanceCount, instanceCount,
                                instanceCount, instanceCount,
                                instanceCount});

    //
    std::array<uint32_t, 2> hLeafSegmentRangeArray = {0u, uint32_t(instanceCount)};
    std::array<uint32_t, 2> hNodeSegmentRangeArray = {0u, std::max(1u, uint32_t(instanceCount - 1))};
    queue.MemcpyAsync(dLeafSegmentRange, Span<const uint32_t>(hLeafSegmentRangeArray));
    queue.MemcpyAsync(dNodeSegmentRange, Span<const uint32_t>(hNodeSegmentRangeArray));

    // Get the scene AABB from the leaf AABBs,
    Reduce(Span<AABB3, 1u>(dSceneAABB.data(), 1u), dTemp,
           ToConstSpan(dLeafAABBs), AABB3::Negative(),
           queue, UnionAABB3Functor());

    // Since we work with AABBs we do not need to store
    // centers but next kernel designed to get primitive center,
    // so temporarily generating prim center. This is somewhat a waste
    // TODO: Change this later maybe?
    queue.IssueWorkKernel<KCGenAABBCenters>
    (
        "KCGenCentersFromAABBs",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(instanceCount)},
        dAABBCenters,
        dLeafAABBs
    );
    //
    {
        uint32_t blockCount = blockPerSegment;
        queue.IssueBlockKernel<KCGenMortonCode>
        (
            "KCGenMortonCodes",
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = StaticThreadPerBlock1D()
            },
            // Output
            dMortonCodes[0],
            // Inputs
            ToConstSpan(dLeafSegmentRange),
            ToConstSpan(dSceneAABB),
            //
            dAABBCenters,
            blockPerSegment
        );
    };
    // Sort
    Iota(dIndices[0], 0u, queue);
    uint32_t sortedIndex = RadixSort<true, uint64_t, uint32_t>
    (
        dMortonCodes, dIndices, dTemp, queue
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
    // Now we have a multiple valid morton code lists,
    // Construct the node hierarchy
    {
        uint32_t blockCount = blockPerSegment;
        queue.IssueBlockKernel<KCConstructLBVHInternalNodes>
        (
            "KCConstructLBVHInternalNodes",
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = StaticThreadPerBlock1D()
            },
            // Output
            dNodes,
            dLeafParentIndices,
            // Inputs
            ToConstSpan(dLeafSegmentRange),
            ToConstSpan(dNodeSegmentRange),
            ToConstSpan(dMortonCodes[0]),
            ToConstSpan(dIndices[0]),
            //
            blockPerSegment,
            1u
        );
    };
    // Finally at AABB union portion now, union the AABBs.
    queue.MemsetAsync(dAtomicCounters, 0x00);
    {
        uint32_t blockCount = blockPerSegment;
        queue.IssueBlockKernel<KCUnionLBVHBoundingBoxes>
        (
            "KCUnionLBVHBoundingBoxes",
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = StaticThreadPerBlock1D()
            },
            // Output
            dBoundingBoxes,
            dAtomicCounters,
            // Inputs
            ToConstSpan(dNodes),
            ToConstSpan(dLeafParentIndices),
            ToConstSpan(dLeafSegmentRange),
            ToConstSpan(dNodeSegmentRange),
            ToConstSpan(dLeafAABBs),
            //
            blockPerSegment,
            1u
        );
    }
    // Issue copy and wait
    LBVHBoundingBox hBBox;
    queue.MemcpyAsync(Span<LBVHBoundingBox>(&hBBox, 1),
                      ToConstSpan(dBoundingBoxes.subspan(0, 1)));
    queue.Barrier().Wait();
    // And done!
    sceneAABB = AABB3(Vector3(hBBox.min[0], hBBox.min[1], hBBox.min[2]),
                      Vector3(hBBox.max[0], hBBox.max[1], hBBox.max[2]));
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
    // We need one more potential partition which is "miss" partition
    maxPartitionCount++;

    rayPartitioner = RayPartitioner(gpuSystem, static_cast<uint32_t>(maxRayCount),
                                    static_cast<uint32_t>(maxPartitionCount));
    MemAlloc::AllocateMultiData(std::tie(dBitStacks, dPrevNodeIndices),
                                stackMem, {maxRayCount, maxRayCount});
}

void BaseAcceleratorLBVH::CastRays(// Output
                                   Span<HitKeyPack> dHitIds,
                                   Span<MetaHit> dHitParams,
                                   // I-O
                                   Span<BackupRNGState> dRNGStates,
                                   Span<RayGMem> dRays,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    assert(maxPartitionCount != 0);
    assert(dRayIndices.size() != 0);
    uint32_t allRayCount = static_cast<uint32_t>(dRays.size());
    uint32_t currentRayCount = static_cast<uint32_t>(dRayIndices.size());
    uint32_t partitionCount = static_cast<uint32_t>(maxPartitionCount);
    // Root node is node index 0, so we are lucky we can memset
    queue.MemsetAsync(dPrevNodeIndices.subspan(0, allRayCount), 0x00);
    // For bit stack however we need to set it old fashioned way)
    queue.IssueWorkKernel<KCInitializeBitStack>
    (
        "KCInitBitStack",
        DeviceWorkIssueParams{.workCount = allRayCount},
        dBitStacks
    );
    // Initialize the ray partitioner
    auto [dCurrentIndices, dCurrentKeys] = rayPartitioner.Start(currentRayCount,
                                                                partitionCount,
                                                                queue);
    // Copy the ray indices to the local buffer, normally we could utilize
    // global ray partitioner (if available) but
    // - Not all renderers (very simple ones probably) may not have a partitioner
    // - OptiX (or equivalent on other hardwares hopefully in the future) already
    //   does two-level acceleration in hardware, so we don't need to do this
    queue.MemcpyAsync(dCurrentIndices, dRayIndices);
    // Continuously do traverse/partition until all rays are missed
    while(currentRayCount != 0)
    {
        queue.IssueWorkKernel<KCIntersectBaseLBVH>
        (
            "(A)LBVHRayCast"sv,
            DeviceWorkIssueParams{.workCount = currentRayCount},
            // Output
            dCurrentKeys,
            // I-O
            dBitStacks,
            dPrevNodeIndices,
            // Input
            ToConstSpan(dRays),
            ToConstSpan(dCurrentIndices),
            // Constants
            ToConstSpan(dLeafKeys),
            ToConstSpan(dLeafAABBs),
            ToConstSpan(dNodes),
            ToConstSpan(dBoundingBoxes)
        );

        static constexpr CommonKey IdBits = AcceleratorKey::IdBits;
        auto batchRange = Vector2ui(IdBits, IdBits + maxBitsUsedOnKey[0]);
        auto idRange = Vector2ui(0, maxBitsUsedOnKey[1]);
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
        dCurrentIndices = dIndices;
        dCurrentKeys = dKeys;

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
                dCurrentKeys = dCurrentKeys.subspan(0, currentRayCount);
                dCurrentIndices = dCurrentIndices.subspan(0, currentRayCount);
            }
            else
            {
                // Normal work, find the accel group issue the kernel
                uint32_t partitionStart = hPartitionOffsets[pIndex];
                uint32_t localSize = hPartitionOffsets[pIndex + 1] - partitionStart;

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
                                          dRNGStates,
                                          dRays,
                                          // Input
                                          dLocalIndices,
                                          dLocalKeys,
                                          //
                                          key.FetchBatchPortion(),
                                          queue);
            }
        }
    }
}

void BaseAcceleratorLBVH::CastVisibilityRays(// Output
                                             Bitspan<uint32_t> dIsVisibleBuffer,
                                             // I-O
                                             Span<BackupRNGState> dRNGStates,
                                             // Input
                                             Span<const RayGMem> dRays,
                                             Span<const RayIndex> dRayIndices,
                                             const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Visibility Casting"sv);
    const auto _ = annotation.AnnotateScope();

    assert(maxPartitionCount != 0);
    assert(dRayIndices.size() != 0);
    uint32_t allRayCount = static_cast<uint32_t>(dRays.size());
    uint32_t currentRayCount = static_cast<uint32_t>(dRayIndices.size());
    uint32_t partitionCount = static_cast<uint32_t>(maxPartitionCount);
    // Root node is node index 0, so we are lucky we can memset
    queue.MemsetAsync(dPrevNodeIndices.subspan(0, allRayCount), 0x00);
    // Assume visible, cull if hits anything
    queue.IssueWorkKernel<KCSetIsVisibleIndirect>
    (
        "KCSetIsVisibleIndirect"sv,
        DeviceWorkIssueParams{.workCount = currentRayCount},
        dIsVisibleBuffer,
        dRayIndices
    );

    // For bit stack however we need to set it old fashioned way)
    queue.IssueWorkKernel<KCInitializeBitStack>
    (
        "KCInitBitStack",
        DeviceWorkIssueParams{.workCount = allRayCount},
        dBitStacks
    );
    // Initialize the ray partitioner
    auto [dCurrentIndices, dCurrentKeys] = rayPartitioner.Start(currentRayCount,
                                                                partitionCount,
                                                                queue);
    // Copy the ray indices to the local buffer, normally we could utilize
    // global ray partitioner (if available) but
    // - Not all renderers (very simple ones probably) may not have a partitioner
    // - OptiX (or equivalent on other hardwares hopefully in the future) already
    //   does two-level acceleration in hardware, so we don't need to do this
    queue.MemcpyAsync(dCurrentIndices, dRayIndices);
    // Continuously do traverse/partition until all rays are missed
    while(currentRayCount != 0)
    {
        queue.IssueWorkKernel<KCIntersectBaseLBVH>
        (
            "(A)LBVHRayCast"sv,
            DeviceWorkIssueParams{.workCount = currentRayCount},
            // Output
            dCurrentKeys,
            // I-O
            dBitStacks,
            dPrevNodeIndices,
            // Input
            ToConstSpan(dRays),
            ToConstSpan(dCurrentIndices),
            // Constants
            ToConstSpan(dLeafKeys),
            ToConstSpan(dLeafAABBs),
            ToConstSpan(dNodes),
            ToConstSpan(dBoundingBoxes)
        );
        static constexpr CommonKey IdBits = AcceleratorKey::IdBits;
        auto batchRange = Vector2ui(IdBits, IdBits + maxBitsUsedOnKey[0]);
        auto idRange = Vector2ui(0, maxBitsUsedOnKey[1]);
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
        dCurrentIndices = dIndices;
        dCurrentKeys = dKeys;

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
                dCurrentKeys = dCurrentKeys.subspan(0, currentRayCount);
                dCurrentIndices = dCurrentIndices.subspan(0, currentRayCount);
            }
            else
            {
                // Normal work, find the accel group issue the kernel
                uint32_t partitionStart = hPartitionOffsets[pIndex];
                uint32_t localSize = hPartitionOffsets[pIndex + 1] - partitionStart;

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
                accelGroup->CastVisibilityRays(// Output
                                               dIsVisibleBuffer,
                                               // I-O
                                               dRNGStates,
                                               dRays,
                                               // Input
                                               dLocalIndices,
                                               dLocalKeys,
                                               //
                                               key.FetchBatchPortion(),
                                               queue);
            }
        }
    }
}

void BaseAcceleratorLBVH::CastLocalRays(// Output
                                        Span<HitKeyPack> dHitIds,
                                        Span<MetaHit> dHitParams,
                                        // I-O
                                        Span<BackupRNGState> dRNGStates,
                                        Span<RayGMem> dRays,
                                        // Input
                                        Span<const RayIndex> dRayIndices,
                                        Span<const AcceleratorKey> dAccelKeys,
                                        CommonKey dAccelKeyBatchPortion,
                                        const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Local Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    auto accelGroupOpt = accelInstances.at(dAccelKeyBatchPortion);
    if(!accelGroupOpt)
    {
        throw MRayError("BaseAccelerator: Unknown accelerator batch {}",
                        dAccelKeyBatchPortion);
    }
    AcceleratorGroupI* accelGroup = accelGroupOpt.value().get();
    auto dAccelKeysCommon = MemAlloc::RepurposeAlloc<const CommonKey>(dAccelKeys);

    accelGroup->CastLocalRays(// Output
                              dHitIds,
                              dHitParams,
                              // I-O
                              dRNGStates,
                              dRays,
                              // Input
                              dRayIndices,
                              dAccelKeysCommon,
                              //
                              dAccelKeyBatchPortion,
                              queue);
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