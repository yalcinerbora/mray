#pragma once

#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgGeneric.h"

#include "TracerTypes.h"
#include "Core/Types.h"

template<uint32_t SELECTION, class UnaryFunc>
class TernaryToBinaryFunctor
{
    UnaryFunc F;

    public:
    TernaryToBinaryFunctor(UnaryFunc f)
        : F(f)
    {}

    MRAY_HYBRID MRAY_CGPU_INLINE
    bool operator()(CommonIndex i) const
    {
        return F(i) == SELECTION;
    }
};

struct MultiPartitionOutput
{
    // This is always host visible
    Span<uint32_t, 1>   hPartitionCount;
    // Prefix-sum buffer of partitions (TotalSize: partition count + 1)
    // These may be either device-only or host visible according to the
    // user
    bool                isHostVisible;
    Span<uint32_t>      hdPartitionStartOffsets;
    Span<CommonKey>     hdPartitionKeys;
    // Per-key partition indirection index
    // These are always device only
    Span<CommonIndex>   dPartitionIndices;
    Span<CommonKey>     dPartitionKeys;
};

struct BinaryPartitionOutput
{
    // Prefix-sum buffer of partitions (TotalSize: 2 + 1)
    Span<uint32_t, 3>   hPartitionStartOffsets;
    // Per-key partition indirection index
    Span<CommonIndex>   dPartitionIndices;

    std::array<Span<CommonIndex>, 2> Spanify() const;
};

struct TernaryPartitionOutput
{
    // Prefix-sum buffer of partitions (TotalSize: 3 + 1)
    Span<uint32_t, 4>   hPartitionStartOffsets;
    // Per-key partition indirection index
    Span<CommonIndex>   dPartitionIndices;

    std::array<Span<CommonIndex>, 3> Spanify() const;
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
    //
    Span<uint32_t>  hPartitionCount;
    // Host Memory
    Span<uint32_t>  hPartitionStartOffsets;
    Span<CommonKey> hPartitionKeys;
    // If user requested only divice visible these will be used
    // These have much faster bandwidth
    Span<uint32_t>  dPartitionStartOffsets;
    Span<CommonKey> dPartitionKeys;

    //
    uint32_t            rayCount;
    uint32_t            maxPartitionCount;
    bool                isResultsInHostVisible;

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
                                  uint32_t maxPartitionCount,
                                  const GPUQueue& queue,
                                  bool isResultsInHostVisible = true);

    template<class UnaryFunc>
    BinaryPartitionOutput   BinaryPartition(Span<const CommonIndex> dIndices,
                                            const GPUQueue& queue,
                                            UnaryFunc&&) const;
    template<class UnaryFunc>
    TernaryPartitionOutput  TernaryPartition(Span<CommonIndex> dIndicesIn,
                                             const GPUQueue& queue,
                                             UnaryFunc&& UnaryF) const;

    MultiPartitionOutput    MultiPartition(Span<CommonKey> dKeysIn,
                                           Span<CommonIndex> dIndicesIn,
                                           const Vector2ui& keyDataBitRange,
                                           const Vector2ui& keyBatchBitRange,
                                           const GPUQueue& queue,
                                           bool onlySortForBatches = false) const;

    size_t                  GPUMemoryUsage() const;
};

template<class T>
Span<T> RayPartitioner::DetermineOutputSpan(const std::array<Span<T>, 2>& doubleBuffer,
                                            Span<const T> checkedSpan)
{
    auto SisterSpan = [s = checkedSpan, db = doubleBuffer](uint32_t from, uint32_t to)
    {
        ptrdiff_t diff = s.data() - db[from].data();
        assert(diff >= 0);
        return db[to].subspan(uint32_t(diff), s.size());
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
    using namespace std::string_view_literals;
    static const auto annotation = system.CreateAnnotation("Ray BinaryPartition"sv);
    const auto _ = annotation.AnnotateScope();

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

template<class UnaryFunc>
TernaryPartitionOutput RayPartitioner::TernaryPartition(Span<CommonIndex> dIndicesIn,
                                                        const GPUQueue& queue,
                                                        UnaryFunc&& UnaryF) const
{
    static constexpr uint32_t FIRST = 0;
    static constexpr uint32_t SECOND = 1;

    using namespace std::string_view_literals;
    static const auto annotation = system.CreateAnnotation("Ray TernaryPartition"sv);
    const auto _ = annotation.AnnotateScope();

    // Determine output buffer
    Span<CommonIndex> dOutput = DetermineOutputSpan(dIndices, ToConstSpan(dIndicesIn));

    hPartitionStartOffsets[0] = 0;
    Span<uint32_t, 1> hPartitionIndex0(hPartitionStartOffsets.subspan(1, 1));
    Span<uint32_t, 1> hPartitionIndex1(hPartitionStartOffsets.subspan(2, 1));
    hPartitionStartOffsets[3] = static_cast<uint32_t>(dIndicesIn.size());

    using TTB0 = TernaryToBinaryFunctor<FIRST, UnaryFunc>;
    DeviceAlgorithms::BinaryPartition(dOutput,
                                      hPartitionIndex0,
                                      dTempMemory,
                                      ToConstSpan(dIndicesIn),
                                      queue,
                                      TTB0(UnaryF));
    queue.Barrier().Wait();

    auto dRightIn = dIndicesIn.subspan(hPartitionIndex0[0],
                                       dIndicesIn.size() - hPartitionIndex0[0]);
    auto dRightOut = dOutput.subspan(hPartitionIndex0[0],
                                     dIndicesIn.size() - hPartitionIndex0[0]);

    // We already partitioned the entire list, just return
    if(dRightIn.size() == 0)
    {
        hPartitionStartOffsets[2] = static_cast<uint32_t>(dIndicesIn.size());
        return TernaryPartitionOutput
        {
            .hPartitionStartOffsets = Span<uint32_t, 4>(hPartitionStartOffsets.subspan(0, 4)),
            .dPartitionIndices = dOutput
        };
    }

    // Copy the right side back to input buffer, so that we can present
    // a single buffer to the user
    queue.MemcpyAsync(dRightIn, ToConstSpan(dRightOut));
    // Again binary partition
    using TTB1 = TernaryToBinaryFunctor<SECOND, UnaryFunc>;
    DeviceAlgorithms::BinaryPartition(dRightOut,
                                      hPartitionIndex1,
                                      dTempMemory,
                                      ToConstSpan(dRightIn),
                                      queue,
                                      TTB1(UnaryF));

    DeviceAlgorithms::InPlaceTransform
    (
        hPartitionIndex1.subspan(0, 1),
        queue,
        [hPartitionIndex0] MRAY_HYBRID(uint32_t& i)
        {
            i += hPartitionIndex0[0];
        }
    );

    return TernaryPartitionOutput
    {
        .hPartitionStartOffsets = Span<uint32_t, 4>(hPartitionStartOffsets.subspan(0, 4)),
        .dPartitionIndices = dOutput
    };
}

inline size_t RayPartitioner::GPUMemoryUsage() const
{
    return deviceMem.Size();
}