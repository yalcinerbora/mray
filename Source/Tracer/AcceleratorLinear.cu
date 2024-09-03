#include "AcceleratorLinear.h"

#include "Device/GPUAlgReduce.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"

#include "Core/Error.hpp"

#include "TypeFormat.h"

// Generic PrimitiveKey copy kernel
// TODO: find a way to put this somewhere proper
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGeneratePrimitiveKeys(MRAY_GRID_CONSTANT const Span<PrimitiveKey> dAllLeafs,
                             //
                             MRAY_GRID_CONSTANT const Span<const PrimRangeArray> dConcretePrimRanges,
                             MRAY_GRID_CONSTANT const Span<const Vector2ui> dConcreteLeafRanges,
                             MRAY_GRID_CONSTANT const uint32_t groupId)
{
    constexpr Vector2ui INVALID_BATCH = Vector2ui(std::numeric_limits<uint32_t>::max());

    KernelCallParams kp;
    uint32_t totalRanges = static_cast<uint32_t>(dConcreteLeafRanges.size());

    // Block-stride Loop
    for(uint32_t blockId = kp.blockId; blockId < totalRanges;
        blockId += kp.gridSize)
    {
        using namespace TracerConstants;
        uint32_t localTid = kp.threadId;
        MRAY_SHARED_MEMORY PrimRangeArray sPrimRanges;
        MRAY_SHARED_MEMORY Vector2ui sConcreteLeafRange;
        if(localTid < MaxPrimBatchPerSurface)
            sPrimRanges[localTid] = dConcretePrimRanges[blockId][localTid];
        if(localTid < Vector2ui::Dims)
            sConcreteLeafRange[localTid] = dConcreteLeafRanges[blockId][localTid];
        BlockSynchronize();

        // Do batch by batch
        // Glorified iota..
        uint32_t offset = 0;
        for(uint32_t i = 0; i < MaxPrimBatchPerSurface; i++)
        {
            if(sPrimRanges[i] == INVALID_BATCH) break;

            Vector2ui currentRange = sPrimRanges[i];
            uint32_t count = currentRange[1] - currentRange[0];
            for(uint32_t j = localTid; j < count; j += kp.blockSize)
            {
                uint32_t writeIndex = sConcreteLeafRange[0] + offset + j;
                uint32_t keyIndexPortion = currentRange[0] + j;
                auto key = PrimitiveKey::CombinedKey(groupId, keyIndexPortion);
                dAllLeafs[writeIndex] = key;
            }
            offset += count;
        }
        assert(offset == sConcreteLeafRange[1] - sConcreteLeafRange[0]);
        // Before writing new shared memory values wait all threads to end
        BlockSynchronize();
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCIntersectBaseLinear(// Output
                           MRAY_GRID_CONSTANT const Span<CommonKey> dAccelKeys,
                           // I-O
                           MRAY_GRID_CONSTANT const Span<uint32_t> dTraverseIndices,
                           // Input
                           MRAY_GRID_CONSTANT const Span<const RayGMem> dRays,
                           MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                           // Constants
                           MRAY_GRID_CONSTANT const Span<const AcceleratorKey> dLeafs,
                           MRAY_GRID_CONSTANT const Span<const AABB3> dAABBs)
{
    assert(dAABBs.size() == dLeafs.size());
    KernelCallParams kp;
    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayIndex index = dRayIndices[i];
        auto [ray, tMM] = RayFromGMem(dRays, index);

        AcceleratorKey foundKey = AcceleratorKey::InvalidKey();
        // Linear search the next instance
        uint32_t startIndex = dTraverseIndices[index];
        uint32_t instanceCount = static_cast<uint32_t>(dAABBs.size());
        for(uint32_t j = startIndex; j < instanceCount; j++)
        {
            AABB3 aabb = dAABBs[j];
            if(ray.IntersectsAABB(aabb.Min(), aabb.Max(), tMM))
            {
                // Stop traversal delegate to the inner accelerator
                foundKey = dLeafs[j];
                // Save the iteration index
                dTraverseIndices[index] = j + 1;
                break;
            }
        }
        dAccelKeys[i] = static_cast<CommonKey>(foundKey);
    }
}

std::string_view BaseAcceleratorLinear::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Linear"sv;
    return BaseAccelTypeName<Name>;
}

AABB3 BaseAcceleratorLinear::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    assert(instanceOffsets.size() == (generatedAccels.size() + 1));

    // Allocate
    size_t instanceCount = instanceOffsets.back();
    MemAlloc::AllocateMultiData(std::tie(dLeafs, dAABBs),
                                accelMem,
                                {instanceCount, instanceCount});
    // Write leafs and transformed aabbs to the array
    size_t i = 0;
    GPUQueueIteratorRoundRobin qIt(gpuSystem);
    for(const auto& accGroup : generatedAccels)
    {
        AcceleratorGroupI* aGroup = accGroup.second.get();
        size_t localCount = instanceOffsets[i + 1] - instanceOffsets[i];
        auto dAABBRegion = dAABBs.subspan(instanceOffsets[i],
                                         localCount);
        auto dLeafRegion = dLeafs.subspan(instanceOffsets[i], localCount);
        aGroup->WriteInstanceKeysAndAABBs(dAABBRegion, dLeafRegion, qIt.Queue());

        i++;
        qIt.Next();
    }

    // Reduce the given AABBs
    // Cheeckly utilize stack mem as temp mem
    size_t tempMemSize = DeviceAlgorithms::ReduceTMSize<AABB3>(dAABBs.size());
    Span<AABB3> dReducedAABB;
    Span<Byte> dTemp;
    MemAlloc::AllocateMultiData(std::tie(dTemp, dReducedAABB),
                                stackMem, {tempMemSize, 1});

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    DeviceAlgorithms::Reduce(Span<AABB3,1>(dReducedAABB), dTemp,
                             ToConstSpan(dAABBs),
                             AABB3::Negative(),
                             queue, UnionAABB3Functor());

    AABB3 hAABB;
    queue.MemcpyAsync(Span<AABB3>(&hAABB, 1), ToConstSpan(dReducedAABB));
    queue.Barrier().Wait();

    // This is very simple "accelerator" so basically we are done
    return hAABB;
}

void BaseAcceleratorLinear::AllocateForTraversal(size_t maxRayCount)
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
    MemAlloc::AllocateMultiData(std::tie(dTraversalStack), stackMem, {maxRayCount});
}

void BaseAcceleratorLinear::CastRays(// Output
                                     Span<HitKeyPack> dHitIds,
                                     Span<MetaHit> dHitParams,
                                     // I-O
                                     Span<BackupRNGState> rngStates,
                                     Span<RayGMem> dRays,
                                     // Input
                                     Span<const RayIndex> dRayIndices,
                                     const GPUQueue& queue)
{
    assert(maxPartitionCount != 0);
    using namespace std::string_view_literals;
    queue.MemsetAsync(dTraversalStack, 0x00);

    //DeviceDebug::DumpGPUMemToFile("dRays", ToConstSpan(dRays), queue);

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
        queue.IssueSaturatingKernel<KCIntersectBaseLinear>
        (
            "(A)LinearRayCast"sv,
            KernelIssueParams{.workCount = static_cast<uint32_t>(dCurrentIndices.size())},
            // Output
            dCurrentKeys,
            // I-O
            dTraversalStack,
            // Input
            dRays,
            dCurrentIndices,
            // Constants
            ToConstSpan(dLeafs),
            ToConstSpan(dAABBs)
        );

        static constexpr CommonKey IdBits = AcceleratorKey::IdBits;
        static constexpr CommonKey BatchBits = AcceleratorKey::BatchBits;
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

void BaseAcceleratorLinear::CastShadowRays(// Output
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

void BaseAcceleratorLinear::CastLocalRays(// Output
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

size_t BaseAcceleratorLinear::GPUMemoryUsage() const
{
    size_t totalSize = accelMem.Size() + stackMem.Size();
    for(const auto& [_, accelGroup] : this->generatedAccels)
    {
        totalSize += accelGroup->GPUMemoryUsage();
    }
    return totalSize;
}