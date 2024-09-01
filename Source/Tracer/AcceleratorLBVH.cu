#include "AcceleratorLBVH.h"

#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgRadixSort.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgGeneric.h"

#include "Core/GraphicsFunctions.h"
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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenMortonCode(MRAY_GRID_CONSTANT const Span<uint64_t> dMortonCodes,
                     // Inputs
                     MRAY_GRID_CONSTANT const Span<const Vector3> dPrimCenters,
                     // Constants
                     MRAY_GRID_CONSTANT const Span<const AABB3, 1> dSceneAABB)
{
    assert(dMortonCodes.size() == dPrimCenters.size());
    uint32_t leafCount = static_cast<uint32_t>(dMortonCodes.size());
    KernelCallParams kp;

    Vector3 sceneSpan = dSceneAABB[0].GeomSpan();
    Float maxSide = sceneSpan[sceneSpan.Maximum()];

    using namespace Graphics::MortonCode;
    static constexpr uint64_t MaxBits = MaxBits3D<uint64_t>();
    static constexpr Float SliceCount = Float(1ull << MaxBits);
    // TODO: Check if 32-bit float is not enough here
    Float deltaRecip = Float(SliceCount) / maxSide;

    for(uint32_t globalId = kp.GlobalId(); globalId < leafCount; globalId += kp.TotalSize())
    {
        Vector3 center = dPrimCenters[globalId];
        // Quantize the center
        center *= deltaRecip;
        center.RoundSelf();
        Vector3ui xyz(center[0], center[1], center[2]);
        dMortonCodes[globalId] = Compose3D<uint64_t>(xyz);
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
        auto [ray, tMM] = RayFromGMem(dRays, index);

        AcceleratorKey foundKey = AcceleratorKey::InvalidKey();
        uint32_t stackState = dBitStackStates[index];
        uint32_t currentNodeIndex = dPrevNodeIndices[index];
        using namespace LBVHAccelDetail;
        BitStack bitStack(FetchSubPortion(stackState, StackStateRange),
                          FetchSubPortion(stackState, DepthRange));
        // Traversal
        uint32_t newPrevIndex = TraverseLBVH(bitStack, tMM, ray, dNodes, true,
                                             currentNodeIndex,
        [&](Vector2& tMM, uint32_t leafIndex)
        {
            AABB3 aabb = dLeafAABBs[leafIndex];
            bool doIntersect = ray.IntersectsAABB(aabb.Min(), aabb.Max(), tMM);
            if(doIntersect)
                foundKey = dLeafKeys[leafIndex];
            // Stop traversal delegate to the inner accelerator
            return doIntersect;
        });

        // Save the traversal stack state
        static constexpr auto StackBits = BaseAcceleratorLBVH::StackBitCount;
        static constexpr auto DepthBits = BaseAcceleratorLBVH::DepthBitCount;
        dBitStackStates[index] = bitStack.CompressState<StackBits, DepthBits>();
        dPrevNodeIndices[index] = newPrevIndex;
        // Return the currently found key
        dAccelKeys[index] = static_cast<CommonKey>(foundKey);
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
    // Assume a complete tree, and allocate conservatively
    size_t conservativeNodeCount = MathFunctions::NextPowerOfTwo(instanceCount) << 1u;
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
    AABB3 sceneAABB = LBVHAccelDetail::MulitBuildLBVH(queue)[0];

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