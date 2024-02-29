#include "RayPartitioner.h"

#include <limits>

#include "Core/BitFunctions.h"

#define INVALID_LOCATION std::numeric_limits<uint32_t>::max()
#define FIND_SPLITS_TPB 512

#ifdef MRAY_GPU_BACKEND_CUDA

#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"
#include "cub/block/block_adjacent_difference.cuh"

template<int TPB>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCFindSplits(//Output
                  const Span<uint32_t> gMarks,
                  // Input
                  const Span<const CommonKey> gSortedKeys,
                  // Constants
                  const Vector2ui batchBitRange)
{
    KernelCallParams kp;

    assert(gMarks.size() == gSortedKeys.size());
    uint32_t locCount = static_cast<uint32_t>(gSortedKeys.size());

    static constexpr int ITEMS_PER_THREAD = 4;
    static constexpr uint32_t DATA_PER_BLOCK = TPB * ITEMS_PER_THREAD;

    using KVPair = cub::KeyValuePair<CommonKey, uint32_t>;
    using AdjDifference = cub::BlockAdjacentDifference<KVPair, TPB>;
    using BlockLoad = cub::BlockLoad<CommonKey, TPB, ITEMS_PER_THREAD,
                                     cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore = cub::BlockStore<uint32_t, TPB, ITEMS_PER_THREAD,
                                       cub::BLOCK_STORE_VECTORIZE>;

    uint32_t totalBlock = (locCount + DATA_PER_BLOCK - 1) / DATA_PER_BLOCK;
    CommonKey predecessor = INVALID_LOCATION;

    // DifferenceOp for Adjacentdifference
    auto FindMark = [&batchBitRange](KVPair current, KVPair prev) -> uint32_t
    {
        bool foundSplit = (BitFunctions::FetchSubPortion(current.key, batchBitRange) !=
                           BitFunctions::FetchSubPortion(prev.key, batchBitRange));
        foundSplit |= (current.value == 0);

        uint32_t mark = (foundSplit) ? current.value : INVALID_LOCATION;
        return mark;
    };

    // Block Loop
    for(uint32_t blockId = kp.blockId;
        blockId < totalBlock;
        blockId += kp.gridSize)
    {
        uint32_t processedItemsSoFar = blockId * DATA_PER_BLOCK;
        uint32_t validItems = min(DATA_PER_BLOCK, locCount - processedItemsSoFar);
        const CommonKey* blockLocalKeys = gSortedKeys.data() + processedItemsSoFar;
        uint32_t* blockLocalMarks = gMarks.data() + processedItemsSoFar;

        CommonKey keys[ITEMS_PER_THREAD];
        if(validItems == DATA_PER_BLOCK) [[likely]]
        {
            BlockLoad().Load(blockLocalKeys, keys);
        }
        else
        {
            BlockLoad().Load(blockLocalKeys, keys, validItems);
        }
        // Load predecessor for non-zero blocks
        if(kp.threadId == 0 && kp.blockId != 0)
            predecessor = blockLocalKeys[-1];

        // Convert globalId and key to pairs, difference operator will use these
        KVPair kvPairs[ITEMS_PER_THREAD];
        UNROLL_LOOP
        for(uint32_t i = 0; i < ITEMS_PER_THREAD; i++)
        {
            kvPairs[i].key = keys[i];
            kvPairs[i].value = processedItemsSoFar + kp.threadId * ITEMS_PER_THREAD + i;
        }

        // Actual Adjacent difference call
        uint32_t marks[ITEMS_PER_THREAD];
        AdjDifference().SubtractLeft(kvPairs, marks, FindMark, KVPair{predecessor, 0});

        // Finally do store
        if(validItems == DATA_PER_BLOCK) [[likely]]
        {
            BlockStore().Store(blockLocalMarks, marks);
        }
        else
        {
            BlockStore().Store(blockLocalMarks, marks, validItems);
        }
        // Barrier here for shared memory
        MRAY_DEVICE_BLOCK_SYNC();
    }
}

#else

template<int TPB>
MRAY_KERNEL //MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCFindSplits(//Output
                  const Span<uint32_t> gMarks,
                  // Input
                  const Span<const CommonKey> gSortedKeys,
                  // Constants
                  const Vector2ui batchBitRange)
{
    KernelCallParams kp;
    assert(gMarks.size() == gSortedKeys.size());
    uint32_t locCount = static_cast<uint32_t>(gSortedKeys.size() - 1);

    for(uint32_t globalId = kp.GlobalId();
        globalId < locCount; globalId += kp.TotalSize())
    {
        HitKey key = HitKey(gSortedKeys[globalId]);
        HitKey keyN = HitKey(gSortedKeys[globalId + 1]);

        // Mark the splits
        bool isSplitFound = (BitFunctions::FetchSubPortion(key, batchBitRange) !=
                             BitFunctions::FetchSubPortion(keyN, batchBitRange));
        uint32_t mark = (isSplitFound) ? (globalId + 1) : INVALID_LOCATION;

        gMarks[globalId + 1] = mark;
    }

    // Init first location also
    if(kp.GlobalId() == 0)
        gMarks[0] = 0;
}

#endif // MRAY_GPU_BACKEND_CUDA


MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCFindBinMatIds(// Output
                     Span<CommonKey> gBinKeys,
                     // I-O
                     Span<uint32_t> gBinRanges,
                     // Input
                     Span<const uint32_t> dDenseSplitIndices,
                     Span<const CommonKey> gSortedHitKeys,
                     Span<const uint32_t, 1> gPartitionCount,
                     // Constants
                     uint32_t totalRays)
{
    KernelCallParams kp;

    const uint32_t& partitionCount = gPartitionCount[0];
    for(uint32_t globalId = kp.GlobalId();
        globalId < partitionCount;
        globalId += kp.TotalSize())
    {
        uint32_t index = dDenseSplitIndices[globalId];
        gBinRanges[globalId] = dDenseSplitIndices[globalId];
        gBinKeys[globalId] = gSortedHitKeys[index];
    }

    if(kp.GlobalId() == 0)
        gBinRanges[partitionCount] = totalRays;
}

size_t PartitionerDeviceBufferSize(size_t maxElementEstimate)
{
    using namespace DeviceAlgorithms;
    size_t radixSortTM = RadixSortTMSize<false, CommonKey, CommonIndex>(maxElementEstimate);
    size_t partitionTM = BinPartitionTMSize<CommonIndex>(maxElementEstimate);
    size_t totalTempMem = std::max(radixSortTM, partitionTM);

    size_t totalBytes = (maxElementEstimate * sizeof(CommonKey) * 2 +
                         maxElementEstimate * sizeof(CommonIndex) * 2 +
                         totalTempMem);
    return totalBytes;
}

size_t PartitionerHostBufferSize(size_t maxPartitionEstimate)
{
    size_t totalBytes = ((maxPartitionEstimate + 1) * sizeof(uint32_t) +
                         maxPartitionEstimate * sizeof(CommonKey) +
                         sizeof(uint32_t));
    return totalBytes;
}

RayPartitioner::RayPartitioner(const GPUSystem& system,
                               uint32_t maxElementEstimate,
                               uint32_t maxPartitionEstimate)
    : system(system)
    , deviceMem(system.AllGPUs(), 32_MiB,
                PartitionerDeviceBufferSize(maxElementEstimate),
                true)
    , hostMem(system,
              PartitionerHostBufferSize(maxPartitionEstimate),
              true)
    // TODO: Change this
    , hPartitionCount(reinterpret_cast<uint32_t*>(static_cast<Byte*>(hostMem)), 1)
    , rayCount(0)
    , maxPartitionCount(0)
{}

RayPartitioner::InitialBuffers RayPartitioner::Start(uint32_t rayCountIn,
                                                     uint32_t maxPartitionCountIn)
{
    rayCount = rayCountIn;
    maxPartitionCount = maxPartitionCountIn;

    size_t tempMemSizeIf = DeviceAlgorithms::BinPartitionTMSize<CommonKey>(rayCount);
    size_t tempMemSizeSort = DeviceAlgorithms::RadixSortTMSize<false, CommonKey, CommonIndex>(rayCount);
    size_t totalTempMemSize = std::max(tempMemSizeIf, tempMemSizeSort);

    MemAlloc::AllocateMultiData(std::tie(dKeys[0], dKeys[1],
                                            dIndices[0], dIndices[1],
                                            dTempMemory),
                                deviceMem,
                                {rayCount, rayCount,
                                rayCount, rayCount,
                                totalTempMemSize});
    dKeys[0] = dKeys[0].subspan(0, rayCount);
    dKeys[1] = dKeys[1].subspan(0, rayCount);
    dIndices[0] = dIndices[0].subspan(0, rayCount);
    dIndices[1] = dIndices[1].subspan(0, rayCount);

    Span<uint32_t> hPartCount;
    MemAlloc::AllocateMultiData(std::tie(hPartitionStartOffsets,
                                         hPartitionKeys,
                                         hPartCount),
                                hostMem,
                                {maxPartitionCount + 1,
                                 maxPartitionCount,
                                 1});

    hPartitionStartOffsets = hPartitionStartOffsets.subspan(0, maxPartitionCount + 1);
    hPartitionKeys = hPartitionKeys.subspan(0, maxPartitionCount);
    hPartitionCount = Span<uint32_t, 1>(hPartCount.subspan(0, 1));

    return InitialBuffers
    {
        .dIndices = dIndices[0],
        .dKeys = dKeys[0]
    };
}

MultiPartitionOutput RayPartitioner::MultiPartition(Span<CommonKey> dKeysIn,
                                                    Span<CommonIndex> dIndicesIn,
                                                    const Vector2ui& keyDataBitRange,
                                                    const Vector2ui& keyBatchBitRange,
                                                    const GPUQueue& queue,
                                                    bool onlySortForBatches) const
{
    using namespace DeviceAlgorithms;
    using namespace MemAlloc;
    static_assert(sizeof(uint32_t) <= sizeof(CommonIndex));

    assert(keyDataBitRange[0] != keyDataBitRange[1]);
    assert(keyBatchBitRange[0] != keyBatchBitRange[1]);
    assert(dKeysIn.size() == dIndicesIn.size());

    uint32_t partitionedRayCount = static_cast<uint32_t>(dKeysIn.size());
    Span<CommonKey> dKeysOut = DetermineOutputSpan(dKeys, ToConstSpan(dKeysIn));
    Span<CommonIndex> dIndicesOut = DetermineOutputSpan(dIndices, ToConstSpan(dIndicesIn));
    // Sort Data portion if requested
    Span<CommonKey> dKeysDB[2] = {dKeysIn, dKeysOut};
    Span<CommonIndex> dIndicesDB[2] = {dIndicesIn, dIndicesOut};
    if(!onlySortForBatches)
    {
        uint32_t outIndex = RadixSort<false>(Span<Span<CommonKey>, 2>(dKeysDB),
                                             Span<Span<CommonIndex>, 2>(dIndicesDB),
                                             dTempMemory,
                                             queue,
                                             keyDataBitRange);
        if(outIndex == 1) std::swap(dKeysDB[0], dKeysDB[1]);
    }
    // Then sort batch portion
    uint32_t outIndex = RadixSort<false>(Span<Span<CommonKey>, 2>(dKeysDB),
                                         Span<Span<CommonIndex>, 2>(dIndicesDB),
                                         dTempMemory,
                                         queue,
                                         keyBatchBitRange);
    if(outIndex == 1) std::swap(dKeysDB[0], dKeysDB[1]);


    // Rename/Repurpose buffers for readability
    Span<CommonIndex> dSortedRayIndices = dIndicesDB[0];
    Span<CommonKey> dSortedKeys = dKeysDB[0];
    Span<uint32_t> dSparseSplitIndices = RepurposeAlloc<uint32_t>(dIndices[1]).subspan(0, partitionedRayCount);
    Span<uint32_t> dDenseSplitIndices = RepurposeAlloc<uint32_t>(dKeys[1]).subspan(0, partitionedRayCount);

    // Mark the split positions
    const void* KCFindSplitsPtr = reinterpret_cast<const void*>(&KCFindSplits<FIND_SPLITS_TPB>);
    uint32_t blockCount = (queue.RecommendedBlockCountPerSM(KCFindSplitsPtr, FIND_SPLITS_TPB, 0) *
                           queue.SMCount());
    queue.IssueExactKernel<KCFindSplits<FIND_SPLITS_TPB>>
    (
        "KCFindSplits",
        KernelExactIssueParams{.gridSize = blockCount, .blockSize = FIND_SPLITS_TPB},
        //
        dSparseSplitIndices,
        ToConstSpan(dSortedKeys),
        keyBatchBitRange
    );

    // Partition to host visible buffer
    DeviceAlgorithms::BinaryPartition
    (
        //hPartitionStartOffsets,
        dDenseSplitIndices,
        hPartitionCount,
        dTempMemory,
        ToConstSpan(dSparseSplitIndices),
        queue,
        [] MRAY_HYBRID(uint32_t id) -> bool
        {
            return (id != INVALID_LOCATION);
        }
    );

    // Mark the split positions
    queue.IssueKernel<KCFindBinMatIds>
    (
        "KCFindBinMatIds",
        KernelIssueParams{.workCount = maxPartitionCount},
        //
        hPartitionKeys,
        hPartitionStartOffsets,
        ToConstSpan(dDenseSplitIndices),
        ToConstSpan(dSortedKeys),
        ToConstSpan(hPartitionCount),
        partitionedRayCount
    );

    return MultiPartitionOutput
    {
        hPartitionStartOffsets,
        hPartitionCount,
        hPartitionKeys,
        dSortedRayIndices,
        dSortedKeys
    };


}
