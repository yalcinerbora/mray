#include "RayPartitioner.h"

#include <limits>

#include "Core/BitFunctions.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgRadixSort.h"

#define INVALID_LOCATION std::numeric_limits<uint32_t>::max()
#define FIND_SPLITS_TPB 512

#ifndef MRAY_GPU_BACKEND_CUDA

#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"
#include "cub/block/block_adjacent_difference.cuh"

template<int TPB>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCFindSplits(//Output
                  MRAY_GRID_CONSTANT const Span<uint32_t> gMarks,
                  // Input
                  MRAY_GRID_CONSTANT const Span<const CommonKey> gSortedKeys,
                  // Constants
                  MRAY_GRID_CONSTANT const Vector2ui batchBitRange)
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
        bool foundSplit = (Bit::FetchSubPortion(current.key, batchBitRange.AsArray()) !=
                           Bit::FetchSubPortion(prev.key, batchBitRange.AsArray()));
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
        BlockSynchronize();
    }
}

#else

template<int TPB>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCFindSplits(//Output
                  MRAY_GRID_CONSTANT const Span<uint32_t> gMarks,
                  // Input
                  MRAY_GRID_CONSTANT const Span<const CommonKey> gSortedKeys,
                  // Constants
                  MRAY_GRID_CONSTANT const Vector2ui batchBitRange)
{
    assert(gMarks.size() == gSortedKeys.size());
    KernelCallParams kp;

    std::array<CommonKey, 2> range = {batchBitRange[0], batchBitRange[1]};
    uint32_t locCount = static_cast<uint32_t>(gSortedKeys.size() - 1);

    for(uint32_t globalId = kp.GlobalId();
        globalId < locCount; globalId += kp.TotalSize())
    {
        CommonKey key = gSortedKeys[globalId];
        CommonKey keyN = gSortedKeys[globalId + 1];
        // Mark the splits
        bool isSplitFound = (Bit::FetchSubPortion(key, range) !=
                             Bit::FetchSubPortion(keyN, range));
        uint32_t mark = (isSplitFound) ? (globalId + 1) : INVALID_LOCATION;

        gMarks[globalId + 1] = mark;
    }

    // Init first location also
    if(kp.GlobalId() == 0)
        gMarks[0] = 0;
}

#endif

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCFindBinMatIds(// Output
                     Span<CommonKey> gBinKeys,
                     // I-O
                     Span<uint32_t> gBinRanges,
                     // Input
                     Span<const uint32_t> gDenseSplitIndices,
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
        uint32_t index = gDenseSplitIndices[globalId];
        gBinKeys[globalId] = gSortedHitKeys[index];
        gBinRanges[globalId] = index;
    }

    if(kp.GlobalId() == 0)
        gBinRanges[partitionCount] = totalRays;
}

std::array<Span<CommonIndex>, 2> BinaryPartitionOutput::Spanify() const
{
    return
    {
        dPartitionIndices.subspan(hPartitionStartOffsets[0],
                                  hPartitionStartOffsets[1] - hPartitionStartOffsets[0]),
        dPartitionIndices.subspan(hPartitionStartOffsets[1],
                                  hPartitionStartOffsets[2] - hPartitionStartOffsets[1])
    };
}
std::array<Span<CommonIndex>, 3> TernaryPartitionOutput::Spanify() const
{
    return
    {
        dPartitionIndices.subspan(hPartitionStartOffsets[0],
                                  hPartitionStartOffsets[1] - hPartitionStartOffsets[0]),
        dPartitionIndices.subspan(hPartitionStartOffsets[1],
                                  hPartitionStartOffsets[2] - hPartitionStartOffsets[1]),
        dPartitionIndices.subspan(hPartitionStartOffsets[2],
                                  hPartitionStartOffsets[3] - hPartitionStartOffsets[2])
    };
}

size_t PartitionerDeviceBufferSize(size_t maxElementEstimate)
{
    using namespace DeviceAlgorithms;
    size_t radixSortTM = RadixSortTMSize<false, CommonKey, CommonIndex>(maxElementEstimate);
    size_t partitionTM = BinPartitionTMSize<CommonIndex>(maxElementEstimate);
    size_t totalTempMem = std::max(radixSortTM, partitionTM);

    static constexpr size_t Alignment = MemAlloc::DefaultSystemAlignment();
    using Math::NextMultiple;

    size_t totalBytes = (NextMultiple(maxElementEstimate * sizeof(CommonKey), Alignment) * 2 +
                         NextMultiple(maxElementEstimate * sizeof(CommonIndex), Alignment) * 2 +
                         NextMultiple(totalTempMem, Alignment));
    return totalBytes;
}

size_t PartitionerHostBufferSize(size_t maxPartitionEstimate)
{
    static constexpr size_t Alignment = MemAlloc::DefaultSystemAlignment();
    using Math::NextMultiple;

    size_t totalBytes = (NextMultiple((maxPartitionEstimate + 1) * sizeof(uint32_t), Alignment) +
                         NextMultiple(maxPartitionEstimate * sizeof(CommonKey), Alignment) +
                         NextMultiple(sizeof(uint32_t), Alignment));
    return totalBytes;
}

RayPartitioner::RayPartitioner(const GPUSystem& system)
    : system(system)
    , deviceMem(system.AllGPUs(), 2_MiB, 2_MiB)
    , hostMem(system, 1_KiB)
    , rayCount(0)
    , maxPartitionCount(0)
    , isResultsInHostVisible(true)
{}

RayPartitioner::RayPartitioner(const GPUSystem& system,
                               uint32_t maxElementEstimate,
                               uint32_t maxPartitionEstimate)
    : system(system)
    , deviceMem(system.AllGPUs(), 16_MiB,
                PartitionerDeviceBufferSize(maxElementEstimate),
                true)
    , hostMem(system,
              PartitionerHostBufferSize(maxPartitionEstimate),
              true)
    // TODO: Change this, we should use MemAlloc::... function to allocate this,
    // but static span mandates value so...
    , hPartitionCount(reinterpret_cast<uint32_t*>(static_cast<Byte*>(hostMem)), 1)
    , rayCount(0)
    , maxPartitionCount(0)
{}

RayPartitioner::RayPartitioner(RayPartitioner&& other)
    : system(other.system)
    , deviceMem(std::move(other.deviceMem))
    , hostMem(std::move(other.hostMem))
    , dKeys(other.dKeys)
    , dIndices(other.dIndices)
    , dTempMemory(other.dTempMemory)
    , hPartitionStartOffsets(other.hPartitionStartOffsets)
    , hPartitionCount(other.hPartitionCount)
    , hPartitionKeys(other.hPartitionKeys)
    , rayCount(other.rayCount)
    , maxPartitionCount(other.maxPartitionCount)
{}

RayPartitioner& RayPartitioner::operator=(RayPartitioner&& other)
{
    assert(this != &other);
    // There should be one system in circulation but just to be sure
    assert(&system == &other.system);
    deviceMem = std::move(other.deviceMem);
    hostMem = std::move(other.hostMem);
    dKeys = other.dKeys;
    dIndices = other.dIndices;
    dKeys = other.dKeys;
    dTempMemory = other.dTempMemory;
    hPartitionStartOffsets = other.hPartitionStartOffsets;
    hPartitionCount = other.hPartitionCount;
    hPartitionKeys = other.hPartitionKeys;
    rayCount = other.rayCount;
    maxPartitionCount = other.maxPartitionCount;
    return *this;
}

RayPartitioner::InitialBuffers RayPartitioner::Start(uint32_t rayCountIn,
                                                     uint32_t maxPartitionCountIn,
                                                     bool isHostVisible)
{
    isResultsInHostVisible = isHostVisible;

    rayCount = rayCountIn;
    maxPartitionCount = maxPartitionCountIn;
    // We may binary/ternary partition, support at least 3
    maxPartitionCount = std::max(maxPartitionCount, 3u);

    size_t tempMemSizeIf = DeviceAlgorithms::BinPartitionTMSize<CommonKey>(rayCount);
    size_t tempMemSizeSort = DeviceAlgorithms::RadixSortTMSize<true, CommonKey, CommonIndex>(rayCount);
    size_t totalTempMemSize = std::max(tempMemSizeIf, tempMemSizeSort);

    if(isResultsInHostVisible)
    {
        MemAlloc::AllocateMultiData(std::tie(dKeys[0], dKeys[1],
                                             dIndices[0], dIndices[1],
                                             dTempMemory),
                                    deviceMem,
                                    {rayCount, rayCount,
                                     rayCount, rayCount,
                                     totalTempMemSize});

        MemAlloc::AllocateMultiData(std::tie(hPartitionCount,
                                             hPartitionKeys,
                                             hPartitionStartOffsets),
                                    hostMem,
                                    {1, maxPartitionCount,
                                    maxPartitionCount + 1});
    }
    else
    {
        MemAlloc::AllocateMultiData(std::tie(dKeys[0], dKeys[1],
                                             dIndices[0], dIndices[1],
                                             dTempMemory, dPartitionKeys,
                                             dPartitionStartOffsets),
                                    deviceMem,
                                    {rayCount, rayCount,
                                     rayCount, rayCount,
                                     totalTempMemSize,
                                     maxPartitionCount,
                                     maxPartitionCount + 1});

        MemAlloc::AllocateMultiData(std::tie(hPartitionCount),
                                    hostMem, {1});
    }

    dKeys[0] = dKeys[0].subspan(0, rayCount);
    dKeys[1] = dKeys[1].subspan(0, rayCount);
    dIndices[0] = dIndices[0].subspan(0, rayCount);
    dIndices[1] = dIndices[1].subspan(0, rayCount);

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
    using namespace std::string_view_literals;
    static const auto annotation = system.CreateAnnotation("Ray N-way Partition"sv);
    const auto _ = annotation.AnnotateScope();

    using namespace DeviceAlgorithms;
    using namespace MemAlloc;
    static_assert(sizeof(uint32_t) <= sizeof(CommonIndex));

    assert(keyBatchBitRange[0] != keyBatchBitRange[1]);
    assert(dKeysIn.size() == dIndicesIn.size());

    uint32_t partitionedRayCount = static_cast<uint32_t>(dKeysIn.size());
    Span<CommonKey> dKeysOut = DetermineOutputSpan(dKeys, ToConstSpan(dKeysIn));
    Span<CommonIndex> dIndicesOut = DetermineOutputSpan(dIndices, ToConstSpan(dIndicesIn));
    // Sort Data portion if requested
    Span<CommonKey> dKeysDB[2] = {dKeysIn, dKeysOut};
    Span<CommonIndex> dIndicesDB[2] = {dIndicesIn, dIndicesOut};

    // TODO: Why are we doing two seperate sorts? Keys almost always should be contiguous.
    // If not we are wasting information space here.
    // So a single pass should suffice maybe? Reason about this more later
    if(!onlySortForBatches)
    {
        assert(keyDataBitRange[0] != keyDataBitRange[1]);
        uint32_t outIndex = RadixSort<true>(Span<Span<CommonKey>, 2>(dKeysDB),
                                            Span<Span<CommonIndex>, 2>(dIndicesDB),
                                            dTempMemory, queue,
                                            keyDataBitRange);
        if(outIndex == 1)
        {
            std::swap(dKeysDB[0], dKeysDB[1]);
            std::swap(dIndicesDB[0], dIndicesDB[1]);
        }
    }
    // Then sort batch portion
    uint32_t outIndex = RadixSort<true>(Span<Span<CommonKey>, 2>(dKeysDB),
                                        Span<Span<CommonIndex>, 2>(dIndicesDB),
                                        dTempMemory, queue,
                                        keyBatchBitRange);
    if(outIndex == 1)
    {
        std::swap(dKeysDB[0], dKeysDB[1]);
        std::swap(dIndicesDB[0], dIndicesDB[1]);
    }


    // Rename/Repurpose buffers for readability
    Span<CommonIndex> dSortedRayIndices = dIndicesDB[0];
    Span<CommonKey> dSortedKeys = dKeysDB[0];
    Span<uint32_t> dSparseSplitIndices = RepurposeAlloc<uint32_t>(dIndicesDB[1]).subspan(0, partitionedRayCount);
    Span<uint32_t> dDenseSplitIndices = RepurposeAlloc<uint32_t>(dKeysDB[1]).subspan(0, partitionedRayCount);

    // Mark the split positions
    static constexpr auto* Kernel = KCFindSplits<FIND_SPLITS_TPB>;
    uint32_t blockCount = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(Kernel),
        FIND_SPLITS_TPB, 0
    );
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
    auto hPartCountStatic = Span<uint32_t, 1>(hPartitionCount.data(), 1);
    DeviceAlgorithms::BinaryPartition
    (
        dDenseSplitIndices,
        hPartCountStatic,
        dTempMemory,
        ToConstSpan(dSparseSplitIndices),
        queue,
        [] MRAY_HYBRID(uint32_t id) -> bool
        {
            return (id != INVALID_LOCATION);
        }
    );
    Span<uint32_t> hdPartitionStartOffsets = (isResultsInHostVisible) ? hPartitionStartOffsets : dPartitionStartOffsets;
    Span<CommonKey> hdPartitionKeys = (isResultsInHostVisible) ? hPartitionKeys : dPartitionKeys;

    // Debug check to find if partition count
    // found out is more than the set maximum.
    // This is debug only since we need to sync with the GPU
    //
    if constexpr(MRAY_IS_DEBUG)
    {
        uint32_t foundPartitionCount;
        if(!isResultsInHostVisible)
        {
            queue.MemcpyAsync(Span<uint32_t>(&foundPartitionCount, 1),
                              Span<const uint32_t>(hPartCountStatic));
            queue.Barrier().Wait();
        }
        else
        {

            queue.Barrier().Wait();
            foundPartitionCount = hPartCountStatic[0];
        }
        assert(foundPartitionCount <= hdPartitionKeys.size());
    }

    // Mark the split positions
    queue.IssueKernel<KCFindBinMatIds>
    (
        "KCFindBinMatIds",
        KernelIssueParams{.workCount = maxPartitionCount},
        //
        hdPartitionKeys,
        hdPartitionStartOffsets,
        ToConstSpan(dDenseSplitIndices),
        ToConstSpan(dSortedKeys),
        ToConstSpan(hPartCountStatic),
        partitionedRayCount
    );

    return MultiPartitionOutput
    {
        hPartCountStatic,
        //
        isResultsInHostVisible,
        hdPartitionStartOffsets,
        hdPartitionKeys,
        //
        dSortedRayIndices,
        dSortedKeys
    };
}