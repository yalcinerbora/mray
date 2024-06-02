#include "AcceleratorLinear.h"
#include "Device/GPUAlgorithms.h"

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
        for(uint32_t i = startIndex; i < instanceCount; i++)
        {
            AABB3 aabb = dAABBs[i];
            if(ray.IntersectsAABB(aabb.Min(), aabb.Max(), tMM))
            {
                // Stop traversal delegate to the inner accelerator
                foundKey = dLeafs[i];
                // Save the iteration index
                dTraverseIndices[index] = i + 1;
                break;
            }
        }
        dAccelKeys[index] = static_cast<CommonKey>(foundKey);
    }
}

struct UnionAABB3
{
    MRAY_GPU MRAY_GPU_INLINE
    AABB3 operator()(const AABB3& l, const AABB3& r) const
    {
        return l.Union(r);
    }
};

std::string_view BaseAcceleratorLinear::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Linear"sv;
    return BaseAccelTypeName<Name>;
}

AABB3 BaseAcceleratorLinear::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    assert(instanceOffsets.size() == generatedAccels.size());

    // Allocate
    size_t instanceCount = instanceOffsets.back();
    MemAlloc::AllocateMultiData(std::tie(dLeafs, dAABBs),
                                accelMem,
                                {instanceCount, instanceCount});
    // Write leafs and transformed aabbs to the array
    size_t i = 0;
    for(const auto& accGroup : generatedAccels)
    {
        AcceleratorGroupI* aGroup = accGroup.second.get();
        size_t localCount = instanceOffsets[i + 1] - instanceOffsets[i];
        auto aabbRegion = dAABBs.subspan(instanceOffsets[i],
                                         localCount);
        auto leafRegions = dLeafs.subspan(instanceOffsets[i], localCount);
        aGroup->WriteInstanceKeysAndAABBs(dAABBs, dLeafs);
        i++;
    }

    // Reduce the given AABBs
    // Cheeckly utilize stack mem as temp mem
    size_t temMemSize = DeviceAlgorithms::ReduceTMSize<AABB3>(dAABBs.size());
    Span<AABB3> dReducedAABB;
    Span<Byte> dTemp;
    MemAlloc::AllocateMultiData(std::tie(dTemp, dReducedAABB),
                                stackMem, {temMemSize, 1});

    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    DeviceAlgorithms::Reduce(Span<AABB3,1>(dReducedAABB), dTemp,
                             ToConstSpan(dAABBs),
                             AABB3::Negative(),
                             queue, UnionAABB3());

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

    rayPartitioner = RayPartitioner(gpuSystem, static_cast<uint32_t>(maxRayCount),
                                    static_cast<uint32_t>(maxPartitionCount));
    MemAlloc::AllocateMultiData(std::tie(dTraversalStack), stackMem, {maxRayCount});
}

void BaseAcceleratorLinear::CastRays(// Output
                                     Span<HitKeyPack> dHitIds,
                                     Span<MetaHit> dHitParams,
                                     Span<SurfaceWorkKey> dWorkKeys,
                                     // I-O
                                     Span<BackupRNGState> rngStates,
                                     // Input
                                     Span<const RayGMem> dRays,
                                     Span<const RayIndex> dRayIndices,
                                     // Constants
                                     const GPUSystem& s)
{
    using namespace std::string_view_literals;
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
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
        queue.IssueSaturatingKernel<KCIntersectBaseLinear>
        (
            "(A)LinearRayCast"sv,
            KernelIssueParams{.workCount = 0},
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
        static constexpr CommonKey BatchBits = AcceleratorKey::IdBits;
        auto batchRange = Vector2ui(BatchBits, BatchBits + maxBitsUsedOnKey[0]);
        auto idRange = Vector2ui(IdBits, IdBits + maxBitsUsedOnKey[1]);
        auto
        [
            hPartitionOffsets,
            hPartitionCount,
            hKeys,
            dIndices,
            dKeys
        ] = rayPartitioner.MultiPartition(dCurrentKeys,
                                          dCurrentIndices,
                                          idRange,
                                          batchRange,
                                          queue, false);
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
                    throw MRayError("BaseAccelerator: Unknown accelerator key {}", key);
                }
                AcceleratorGroupI* accelGroup = accelGroupOpt.value().get();
                accelGroup->CastLocalRays(// Output
                                          dHitIds,
                                          dHitParams,
                                          dWorkKeys,
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
                                           // Constants
                                           const GPUSystem& s)
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
                                          // Constants
                                          const GPUSystem& s)
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