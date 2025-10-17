#include "Core/MathForward.h"
#include "RayPartitioner.h"

#include <limits>

#include "Core/BitFunctions.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"     // IWYU pragma: keep
#include "Device/GPUAlgRadixSort.h"
#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgGeneric.h"

static constexpr auto INVALID_LOCATION = std::numeric_limits<uint32_t>::max();
static constexpr auto FIND_SPLITS_TPB = 512u;

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

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCFindBinMatIds(// Output
                     MRAY_GRID_CONSTANT const Span<CommonKey> gBinKeys,
                     // I-O
                     MRAY_GRID_CONSTANT const Span<uint32_t> gBinRanges,
                     // Input
                     MRAY_GRID_CONSTANT const Span<const uint32_t> gDenseSplitIndices,
                     MRAY_GRID_CONSTANT const Span<const CommonKey> gSortedHitKeys,
                     MRAY_GRID_CONSTANT const Span<const uint32_t, 1> gPartitionCount,
                     // Constants
                     MRAY_GRID_CONSTANT const uint32_t totalRays)
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

size_t PartitionerDeviceBufferSize(size_t maxElementEstimate, const GPUSystem& gpuSystem)
{
    using namespace DeviceAlgorithms;
    size_t radixSortTM = 0u, partitionTM = 0u;
    for(const auto& gpu : gpuSystem.SystemDevices())
    {
        radixSortTM = Math::Max(RadixSortTMSize<false, CommonKey, CommonIndex>(maxElementEstimate,
                                                                               gpu.GetComputeQueue(0)),
                               radixSortTM);
        partitionTM = Math::Max(BinPartitionTMSize<CommonIndex>(maxElementEstimate,
                                                                gpu.GetComputeQueue(0)),
                               partitionTM);
    }

    size_t totalTempMem = Math::Max(radixSortTM, partitionTM);
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
                PartitionerDeviceBufferSize(maxElementEstimate, system),
                true)
    , hostMem(system,
              PartitionerHostBufferSize(maxPartitionEstimate),
              true)
    // TODO: Change this, we should use MemAlloc::... function to allocate this,
    // but static span mandates value so...
    , hPartitionCount(reinterpret_cast<uint32_t*>(static_cast<Byte*>(hostMem)), 1)
    , rayCount(0)
    , maxPartitionCount(0)
    , isResultsInHostVisible(true)
{}

RayPartitioner::RayPartitioner(RayPartitioner&& other)
    : system(other.system)
    , deviceMem(std::move(other.deviceMem))
    , hostMem(std::move(other.hostMem))
    , dKeys(other.dKeys)
    , dIndices(other.dIndices)
    , dTempMemory(other.dTempMemory)
    , hPartitionCount(other.hPartitionCount)
    , hPartitionStartOffsets(other.hPartitionStartOffsets)
    , hPartitionKeys(other.hPartitionKeys)
    , dPartitionStartOffsets(other.dPartitionStartOffsets)
    , dPartitionKeys(other.dPartitionKeys)
    , rayCount(other.rayCount)
    , maxPartitionCount(other.maxPartitionCount)
    , isResultsInHostVisible(other.isResultsInHostVisible)
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
    dTempMemory = other.dTempMemory;
    hPartitionCount = other.hPartitionCount;
    hPartitionStartOffsets = other.hPartitionStartOffsets;
    hPartitionKeys = other.hPartitionKeys;
    dPartitionStartOffsets = other.dPartitionStartOffsets;
    dPartitionKeys = other.dPartitionKeys;
    rayCount = other.rayCount;
    maxPartitionCount = other.maxPartitionCount;
    isResultsInHostVisible = other.isResultsInHostVisible;
    return *this;
}

RayPartitioner::InitialBuffers RayPartitioner::Start(uint32_t rayCountIn,
                                                     uint32_t maxPartitionCountIn,
                                                     const GPUQueue& queue,
                                                     bool isHostVisible)
{
    isResultsInHostVisible = isHostVisible;

    rayCount = rayCountIn;
    maxPartitionCount = maxPartitionCountIn;
    // We may binary/ternary partition, support at least 3
    maxPartitionCount = Math::Max(maxPartitionCount, 3u);

    size_t tempMemSizeIf = DeviceAlgorithms::BinPartitionTMSize<CommonKey>(rayCount, queue);
    size_t tempMemSizeSort = DeviceAlgorithms::RadixSortTMSize<true, CommonKey, CommonIndex>(rayCount, queue);
    size_t totalTempMemSize = Math::Max(tempMemSizeIf, tempMemSizeSort);

    if(isResultsInHostVisible)
    {
        MemAlloc::AllocateMultiData(Tie(dKeys[0], dKeys[1],
                                        dIndices[0], dIndices[1],
                                        dTempMemory),
                                    deviceMem,
                                    {rayCount, rayCount,
                                     rayCount, rayCount,
                                     totalTempMemSize});

        MemAlloc::AllocateMultiData(Tie(hPartitionCount,
                                        hPartitionKeys,
                                        hPartitionStartOffsets),
                                    hostMem,
                                    {1, maxPartitionCount,
                                    maxPartitionCount + 1});
    }
    else
    {
        MemAlloc::AllocateMultiData(Tie(dKeys[0], dKeys[1],
                                        dIndices[0], dIndices[1],
                                        dTempMemory, dPartitionKeys,
                                        dPartitionStartOffsets),
                                    deviceMem,
                                    {rayCount, rayCount,
                                     rayCount, rayCount,
                                     totalTempMemSize,
                                     maxPartitionCount,
                                     maxPartitionCount + 1});

        MemAlloc::AllocateMultiData(Tie(hPartitionCount),
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
    static const auto annotation = system.CreateAnnotation("Ray N-way Partition");
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

    // TODO: Why are we doing two separate sorts? Keys almost always should be contiguous.
    // If not we are wasting information space here.
    // So a single pass should suffice maybe? Reason about this more later
    auto IssueSort = [&](const Vector2ui& sortRange)
    {
        uint32_t outIndex = RadixSort<true>(Span<Span<CommonKey>, 2>(dKeysDB),
                                            Span<Span<CommonIndex>, 2>(dIndicesDB),
                                            dTempMemory, queue,
                                            sortRange);
        if(outIndex == 1)
        {
            std::swap(dKeysDB[0], dKeysDB[1]);
            std::swap(dIndicesDB[0], dIndicesDB[1]);
        }
    };
    if(onlySortForBatches)
    {
        assert(keyBatchBitRange[0] != keyBatchBitRange[1]);
        // Then sort batch portion
        IssueSort(keyBatchBitRange);
    }
    // We sort both for batches and data, but if ket range
    // is contiguous we can get away with a single sort
    else if(keyDataBitRange[1] == keyBatchBitRange[0])
    {
        Vector2ui keyFullRange(keyDataBitRange[0],
                               keyBatchBitRange[1]);
        assert(keyFullRange[0] != keyFullRange[1]);
        IssueSort(keyFullRange);
    }
    else
    {
        assert(keyBatchBitRange[0] != keyBatchBitRange[1]);
        assert(keyDataBitRange[0] != keyDataBitRange[1]);
        IssueSort(keyDataBitRange);
        IssueSort(keyBatchBitRange);
    }
    // Rename/Repurpose buffers for readability
    Span<CommonIndex> dSortedRayIndices = dIndicesDB[0];
    Span<CommonKey> dSortedKeys = dKeysDB[0];
    Span<uint32_t> dSparseSplitIndices = RepurposeAlloc<uint32_t>(dIndicesDB[1]).subspan(0, partitionedRayCount);
    Span<uint32_t> dDenseSplitIndices = RepurposeAlloc<uint32_t>(dKeysDB[1]).subspan(0, partitionedRayCount);

    // Mark the split positions
    uint32_t segmentCount = static_cast<uint32_t>(dSortedKeys.size());
    uint32_t blockCount = Math::DivideUp(segmentCount, FIND_SPLITS_TPB);
    queue.IssueBlockKernel<KCFindSplits<FIND_SPLITS_TPB>>
    (
        "KCFindSplits",
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = FIND_SPLITS_TPB
        },
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
    queue.IssueWorkKernel<KCFindBinMatIds>
    (
        "KCFindBinMatIds",
        DeviceWorkIssueParams{.workCount = maxPartitionCount},
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