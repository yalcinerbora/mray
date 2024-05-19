#include "AcceleratorLinear.h"

MRAY_KERNEL
void KCIntersectBaseLinear(// Output
                           Span<CommonKey> dAccelKeys,
                           // Input
                           Span<const RayGMem> dRays,
                           Span<const RayIndex> dRayIndices,
                           // Constants
                           size_t rayCount)
{
    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId(); globalId < rayCount;
        globalId += kp.TotalSize())
    {

    }
}

std::string_view BaseAcceleratorLinear::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Linear"sv;
    return BaseAccelTypeName<Name>;
}

void BaseAcceleratorLinear::InternalConstruct(const std::vector<size_t>& instanceOffsets)
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
    // This is very simple "accelerator" so basically we are done
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
            // Input
            dRays,
            dCurrentIndices,
            // Constants
            currentRayCount
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
                AcceleratorGroupI* accelGroup = accelInstances.at(key.FetchBatchPortion());
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