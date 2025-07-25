#include "AcceleratorLinear.h"

#include "Device/GPUAlgReduce.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"

#include "TypeFormat.h"

// Generic PrimitiveKey copy kernel
// TODO: find a way to put this somewhere proper
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGeneratePrimitiveKeys(MRAY_GRID_CONSTANT const Span<PrimitiveKey> dAllLeafs,
                             //
                             MRAY_GRID_CONSTANT const Span<const PrimRangeArray> dConcretePrimRanges,
                             MRAY_GRID_CONSTANT const Span<const Vector2ui> dConcreteLeafRanges,
                             MRAY_GRID_CONSTANT const CommonKey groupId)
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
        #ifdef MRAY_GPU_BACKEND_CPU
            if(localTid == 0)
            {
                sPrimRanges = dConcretePrimRanges[blockId];
                sConcreteLeafRange = dConcreteLeafRanges[blockId];
            }
        #else
            assert(kp.blockSize >= MaxPrimBatchPerSurface);
            assert(kp.blockSize >= Vector2ui::Dims);
            if(localTid < MaxPrimBatchPerSurface)
                sPrimRanges[localTid] = dConcretePrimRanges[blockId][localTid];
            if(localTid < Vector2ui::Dims)
                sConcreteLeafRange[localTid] = dConcreteLeafRanges[blockId][localTid];
        #endif
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

extern MRAY_KERNEL
void KCSetIsVisibleIndirect(MRAY_GRID_CONSTANT const Bitspan<uint32_t> dIsVisibleBuffer,
                            //
                            MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    uint32_t rayCount = uint32_t(dRayIndices.size());
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        dIsVisibleBuffer.SetBitParallel(dRayIndices[i], true);
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
    // Wait all queues
    gpuSystem.SyncAll();

    // Reduce the given AABBs
    // Cheekily utilize stack mem as temp mem
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    size_t tempMemSize = DeviceAlgorithms::ReduceTMSize<AABB3>(dAABBs.size(), queue);
    Span<AABB3> dReducedAABB;
    Span<Byte> dTemp;
    MemAlloc::AllocateMultiData(std::tie(dTemp, dReducedAABB),
                                stackMem, {tempMemSize, 1});

    DeviceAlgorithms::Reduce(Span<AABB3,1>(dReducedAABB), dTemp,
                             ToConstSpan(dAABBs),
                             AABB3::Negative(),
                             queue, UnionAABB3Functor());

    AABB3 hAABB;
    queue.MemcpyAsync(Span<AABB3>(&hAABB, 1), ToConstSpan(dReducedAABB));
    queue.Barrier().Wait();
    // This is very simple "accelerator" so basically we are done
    sceneAABB = hAABB;
    return sceneAABB;
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
                                     Span<BackupRNGState> dRNGStates,
                                     Span<RayGMem> dRays,
                                     // Input
                                     Span<const RayIndex> dRayIndices,
                                     const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    assert(maxPartitionCount != 0);
    assert(dRayIndices.size() != 0);
    uint32_t allRayCount = static_cast<uint32_t>(dRays.size());
    uint32_t currentRayCount = static_cast<uint32_t>(dRayIndices.size());
    uint32_t partitionCount = static_cast<uint32_t>(maxPartitionCount);
    //
    queue.MemsetAsync(dTraversalStack.subspan(0, allRayCount), 0x00);
    // Initialize the ray partitioner
    auto [dCurrentIndices, dCurrentKeys] = rayPartitioner.Start(currentRayCount,
                                                                partitionCount,
                                                                queue);
    // Copy the ray indices to the local buffer, normally we could utilize
    // global ray partitioner (if available) but
    // - Not all renderers (very simple ones probably) may not have a partitioner
    // - OptiX (or equivalent on other hardwares hopefully in the future) already
    //   does two-level acceleration in hardware, so we dont need to do this
    queue.MemcpyAsync(dCurrentIndices, dRayIndices);
    // Continuously do traverse/partition until all rays are missed
    while(currentRayCount != 0)
    {
        queue.IssueWorkKernel<KCIntersectBaseLinear>
        (
            "(A)LinearRayCast"sv,
            DeviceWorkIssueParams{.workCount = currentRayCount},
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

void BaseAcceleratorLinear::CastVisibilityRays(// Output
                                               Bitspan<uint32_t> dIsVisibleBuffer,
                                               // I-O
                                               Span<BackupRNGState> dRNGStates,
                                               // Input
                                               Span<const RayGMem> dRays,
                                               Span<const RayIndex> dRayIndices,
                                               const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Visibility Casting"sv);
    const auto _ = annotation.AnnotateScope();

    assert(maxPartitionCount != 0);
    assert(dRayIndices.size() != 0);
    uint32_t allRayCount = static_cast<uint32_t>(dRays.size());
    uint32_t currentRayCount = static_cast<uint32_t>(dRayIndices.size());
    uint32_t partitionCount = static_cast<uint32_t>(maxPartitionCount);
    //
    queue.MemsetAsync(dTraversalStack.subspan(0, allRayCount), 0x00);
    // Assume visible, cull if hits anything
    queue.IssueWorkKernel<KCSetIsVisibleIndirect>
    (
        "KCSetIsVisibleIndirect"sv,
        DeviceWorkIssueParams{.workCount = currentRayCount},
        dIsVisibleBuffer,
        dRayIndices
    );
    // Initialize the ray partitioner
    auto [dCurrentIndices, dCurrentKeys] = rayPartitioner.Start(currentRayCount,
                                                                partitionCount,
                                                                queue);
    // Copy the ray indices to the local buffer, normally we could utilize
    // global ray partitioner (if available) but
    // - Not all renderers (very simple ones probably) may not have a partitioner
    // - OptiX (or equivalent on other hardwares hopefully in the future) already
    //   does two-level acceleration in hardware, so we dont need to do this
    queue.MemcpyAsync(dCurrentIndices, dRayIndices);
    // Continuously do traverse/partition until all rays are missed
    while(currentRayCount != 0)
    {
        queue.IssueWorkKernel<KCIntersectBaseLinear>
        (
            "(A)LinearRayCast"sv,
            DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dCurrentIndices.size())},
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
                                               // Input
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

void BaseAcceleratorLinear::CastLocalRays(// Output
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
    static const auto annotation = gpuSystem.CreateAnnotation("Local Ray Casting"sv);
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

size_t BaseAcceleratorLinear::GPUMemoryUsage() const
{
    size_t totalSize = accelMem.Size() + stackMem.Size();
    for(const auto& [_, accelGroup] : this->generatedAccels)
    {
        totalSize += accelGroup->GPUMemoryUsage();
    }
    return totalSize;
}