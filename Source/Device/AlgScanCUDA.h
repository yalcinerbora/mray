#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/block/block_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

// Direct wrappers over CUB at the moment
// Probably be refactored later
// Add as you need
namespace mray::cuda::algorithms
{
    template <class T, class BinaryOp>
    MRAY_HOST
    void InclusiveMultiScan(Span<T> dScannedValues,
                            Span<const T> dValues,
                            uint32_t segmentSize,
                            const T& identityElement,
                            const GPUQueueCUDA& queue,
                            BinaryOp&& op);
}

namespace mray::cuda::algorithms
{
    static constexpr uint32_t TPB = StaticThreadPerBlock1D();

    template <class T, class BinaryOp>
    MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
    void KCInclusiveMultiScan(Span<T> dOut,
                              Span<const T> dIn,
                              uint32_t segmentSize,
                              uint32_t totalBlocks,
                              T identityElement,
                              BinaryOp op)
    {
        KernelCallParamsCUDA kp;

        static constexpr uint32_t ITEMS_PER_THREAD = 4;
        static constexpr uint32_t DATA_PER_BLOCK = TPB * ITEMS_PER_THREAD;

        using BlockLoad = cub::BlockLoad<T, TPB, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
        using BlockStore = cub::BlockStore<T, TPB, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;
        using BlockScan = cub::BlockScan<T, TPB>;

        T aggregate = identityElement;
        auto PrefixLoader = [&](T iterationAggregate)
        {
            T temp = aggregate;
            aggregate += iterationAggregate;
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
                MRAY_DEVICE_BLOCK_SYNC();
            }
            aggregate = identityElement;
        }
    }

    template <class T, class BinaryOp>
    MRAY_HOST inline
    void InclusiveMultiScan(Span<T> dScannedValues,
                            Span<const T> dValues,
                            uint32_t segmentSize,
                            const T& identityElement,
                            const GPUQueueCUDA& queue,
                            BinaryOp&& op)
    {
        using namespace std::literals;
        assert(dValues.size() % segmentSize == 0);

        const void* kernelPtr = reinterpret_cast<const void*>(&KCInclusiveMultiScan<T, BinaryOp>);
        uint32_t gridSize = queue.SMCount() * queue.RecommendedBlockCountPerSM(kernelPtr, TPB, 0);
        uint32_t totalBlocks = static_cast<uint32_t>(dValues.size() / segmentSize);

        queue.IssueExactKernel<KCInclusiveMultiScan<T, BinaryOp>>
        (
            "KCMultExclusiveScan"sv,
            KernelExactIssueParams{.gridSize = gridSize, .blockSize = TPB},
            //
            dScannedValues,
            dValues,
            segmentSize,
            totalBlocks,
            identityElement,
            std::forward<BinaryOp>(op)
        );
    }
}