#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemHIP.h"

#include <rocprim/block/block_scan.hpp>
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/device/device_scan.hpp>

namespace mray::hip::algorithms
{

static constexpr uint32_t TPB = StaticThreadPerBlock1D();

template <class T, class BinaryOp>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCInclusiveSegmentedScan(Span<T> dOut,
                              Span<const T> dIn,
                              uint32_t segmentSize,
                              uint32_t totalBlocks,
                              T identityElement,
                              BinaryOp op)
{
    KernelCallParamsHIP kp;

    static constexpr uint32_t ITEMS_PER_THREAD = 4;
    static constexpr uint32_t DATA_PER_BLOCK = TPB * ITEMS_PER_THREAD;

    using BlockLoad = rocprim::BlockLoad<T, TPB, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockStore = rocprim::BlockStore<T, TPB, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
    using BlockScan = rocprim::BlockScan<T, TPB>;

    T aggregate = identityElement;
    auto PrefixLoader = [&](T iterationAggregate)
    {
        T temp = aggregate;
        aggregate = op(aggregate, iterationAggregate);
        return temp;
    };

    // Block-stride loop
    for(uint32_t block = kp.blockId; block < totalBlocks; block += kp.gridSize)
    {
        auto dBlockIn = dIn.subspan(block * segmentSize, segmentSize);
        auto dBlockOut = dOut.subspan(block * segmentSize, segmentSize);

        uint32_t processedItemsSoFar = 0;
        while(processedItemsSoFar != segmentSize)
        {
            uint32_t validItems = min(DATA_PER_BLOCK, segmentSize - processedItemsSoFar);
            auto dSubBlockIn = dBlockIn.subspan(processedItemsSoFar, validItems);
            auto dSubBlockOut = dBlockOut.subspan(processedItemsSoFar, validItems);

            // Load
            T dataRegisters[ITEMS_PER_THREAD];
            if(validItems == DATA_PER_BLOCK) [[likely]]
                BlockLoad().Load(dSubBlockIn.data(), dataRegisters);
            else
                BlockLoad().Load(dSubBlockIn.data(), dataRegisters,
                                 validItems, identityElement);

            // Scan
            BlockScan().InclusiveScan(dataRegisters, dataRegisters,
                                      op, PrefixLoader);

            // Store
            if(validItems == DATA_PER_BLOCK) [[likely]]
                BlockStore().Store(dSubBlockOut.data(), dataRegisters);
            else
                BlockStore().Store(dSubBlockOut.data(), dataRegisters, validItems);

            processedItemsSoFar += validItems;
            BlockSynchronize();
        }
        aggregate = identityElement;
    }
}

template <class T>
MRAY_HOST
size_t ExclusiveScanTMSize(size_t elementCount, const GPUQueueHIP& q)
{
    using namespace rocprim;
    T* dOut = nullptr;
    T* dIn = nullptr;
    void* dTM = nullptr;
    size_t result;
    HIP_CHECK(DeviceScan::ExclusiveScan(dTM, result, dIn, dOut,
                                         [] MRAY_HYBRID(T, T)->T{return T{};},
                                         T{}, static_cast<int>(elementCount)),
                                         ToHandleHIP(q));
    return result;
}

template <class T, class BinaryOp>
MRAY_HOST inline
void InclusiveSegmentedScan(Span<T> dScannedValues,
                            Span<const T> dValues,
                            uint32_t segmentSize,
                            const T& identityElement,
                            const GPUQueueHIP& queue,
                            BinaryOp&& op)
{
    using namespace std::literals;
    assert(dValues.size() % segmentSize == 0);

    uint32_t gridSize = queue.RecommendedBlockCountDevice
    (
        reinterpret_cast<const void*>(&KCInclusiveSegmentedScan<T, BinaryOp>),
        TPB, 0
    );

    uint32_t totalBlocks = static_cast<uint32_t>(dValues.size() / segmentSize);
    queue.IssueBlockKernel<KCInclusiveSegmentedScan<T, BinaryOp>>
    (
        "KCInclusiveSegmentedScan"sv,
        DeviceBlockIssueParams{.gridSize = gridSize, .blockSize = TPB},
        //
        dScannedValues,
        dValues,
        segmentSize,
        totalBlocks,
        identityElement,
        std::forward<BinaryOp>(op)
    );
}

template <class T, class BinaryOp>
MRAY_HOST
void ExclusiveScan(Span<T> dScannedValues,
                   Span<Byte> dTempMem,
                   Span<const T> dValues,
                   const T& initialValue,
                   const GPUQueueHIP& queue,
                   BinaryOp&& op)
{
    using namespace cub;
    assert(dScannedValues.size() == 1 + dValues.size() ||
           dScannedValues.size() == dValues.size());
    size_t tmSize = dTempMem.size();
    CUDA_CHECK(DeviceScan::ExclusiveScan(dTempMem.data(), tmSize,
                                         dValues.data(), dScannedValues.data(),
                                         std::forward<BinaryOp>(op), initialValue,
                                         static_cast<int>(dScannedValues.size()),
                                         ToHandleCUDA(queue)));
}

}