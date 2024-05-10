#pragma once

#include "Device/GPUSystem.h"
#include "Device/GPUAlgorithms.h"

#include "TracerTypes.h"
#include "Core/Types.h"

struct MultiPartitionOutput
{
    // Prefix-sum buffer of partitions (TotalSize: partition count + 1)
    Span<uint32_t>      hPartitionStartOffsets;
    Span<uint32_t, 1>   hPartitionCount;
    Span<CommonKey>     hPartitionKeys;
    // Per-key partition indirection index
    Span<CommonIndex>   dPartitionIndices;
    Span<CommonKey>     dPartitionKeys;
};

struct BinaryPartitionOutput
{
    // Prefix-sum buffer of partitions (TotalSize: 2 + 1)
    Span<uint32_t, 3>   hPartitionStartOffsets;
    // Per-key partition indirection index
    Span<CommonIndex>   dPartitionIndices;
};

class RayPartitioner
{
    public:
    struct InitialBuffers
    {
        Span<CommonIndex> dIndices;
        Span<CommonKey> dKeys;
    };

    private:
    const GPUSystem&    system;
    DeviceMemory        deviceMem;
    HostLocalMemory     hostMem;

    // Device Memory
    std::array<Span<CommonKey>, 2>      dKeys;
    std::array<Span<CommonIndex>, 2>    dIndices;
    Span<Byte>                          dTempMemory;
    // Host Memory
    Span<uint32_t>      hPartitionStartOffsets;
    Span<uint32_t, 1>   hPartitionCount;
    Span<CommonKey>     hPartitionKeys;
    //
    uint32_t            rayCount;
    uint32_t            maxPartitionCount;

    template<class T>
    static Span<T>      DetermineOutputSpan(const std::array<Span<T>,2>& doubleBuffer,
                                            Span<const T> checkedSpan);

    public:
    // Constructors & Destructor
                            RayPartitioner(const GPUSystem& system);
                            RayPartitioner(const RayPartitioner&) = delete;
                            RayPartitioner(RayPartitioner&&);
    RayPartitioner&         operator=(const RayPartitioner&) = delete;
    RayPartitioner&         operator=(RayPartitioner&&);
                            RayPartitioner(const GPUSystem& system,
                                           uint32_t maxElements,
                                           uint32_t maxPartitionCount);

    InitialBuffers          Start(uint32_t rayCount,
                                  uint32_t maxPartitionCount);

    template<class UnaryFunc>
    BinaryPartitionOutput   BinaryPartition(Span<const CommonIndex> dIndices,
                                            const GPUQueue& queue,
                                            UnaryFunc&&) const;

    MultiPartitionOutput    MultiPartition(Span<CommonKey> dKeysIn,
                                           Span<CommonIndex> dIndicesIn,
                                           const Vector2ui& keyDataBitRange,
                                           const Vector2ui& keyBatchBitRange,
                                           const GPUQueue& queue,
                                           bool onlySortForBatches = false) const;
};

template<class T>
Span<T> RayPartitioner::DetermineOutputSpan(const std::array<Span<T>, 2>& doubleBuffer,
                                            Span<const T> checkedSpan)
{
    auto SisterSpan = [s = checkedSpan, db = doubleBuffer](uint32_t from, uint32_t to)
    {
        ptrdiff_t diff = s.data() - db[from].data();
        return db[to].subspan(diff, s.size());
    };

    if(IsSubspan(checkedSpan, doubleBuffer[0]))
        return SisterSpan(0, 1);
    else if(IsSubspan(checkedSpan, doubleBuffer[1]))
        return SisterSpan(1, 0);
    else
        throw MRayError("Ray partitioner is given a unknown span!");
}

template<class UnaryFunc>
BinaryPartitionOutput RayPartitioner::BinaryPartition(Span<const CommonIndex> dIndicesIn,
                                                      const GPUQueue& queue,
                                                      UnaryFunc&& UnaryF) const
{
    // Determine output buffer
    Span<CommonIndex> dOutput = DetermineOutputSpan(dIndices, dIndicesIn);

    hPartitionStartOffsets[0] = 0;
    Span<uint32_t, 1> hPartitionIndex(hPartitionStartOffsets.subspan(1, 1));
    hPartitionStartOffsets[2] = static_cast<uint32_t>(dIndicesIn.size());

    DeviceAlgorithms::BinaryPartition(dOutput,
                                      hPartitionIndex,
                                      dTempMemory,
                                      dIndicesIn,
                                      queue,
                                      std::move(UnaryF));

    return BinaryPartitionOutput
    {
        .hPartitionStartOffsets = Span<uint32_t, 3>(hPartitionStartOffsets.subspan(0, 3)),
        .dPartitionIndices = dOutput
    };
}