#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

struct alignas(std::hardware_destructive_interference_size) LRCounter
{
    uint32_t left;
    uint32_t right;
};

template <class T>
MRAY_HOST inline
size_t BinPartitionTMSize(size_t elementCount, const GPUQueueCPU& q)
{
    uint32_t blockCount = q.DetermineGridStrideBlock(nullptr, 0u,
                                                     StaticThreadPerBlock1D(),
                                                     uint32_t(elementCount));
    return sizeof(LRCounter) * (blockCount + 1);
}

template <class T, class UnaryOp>
MRAY_HOST inline
void BinaryPartition(Span<T> dOutput,
                     Span<uint32_t, 1> dEndOffset,
                     Span<Byte> dTempMemory,
                     Span<const T> dInput,
                     const GPUQueueCPU& queue,
                     UnaryOp&& op)
{
    // Important!!! Binary partition MUST be stable
    // at least the true(left) part
    //
    // Alogrihm Summary:
    //   ---- O(2N + C) where C is the core count.
    //   ---- O(C) memory (two 32-bit integers per core (should be size_t)
    //        aligned to prevent cache flushes or data persistance between threads.
    //
    //
    // For both parths the partition is stable. Each thread
    // compares the element via "UnaryOP". Increments either left
    // or right counters (left side is true side). One comparison
    // for each element.
    // We scan the counters to find the global offsets.
    // We do the comparison again and write according to the offset.
    //
    // Algorithm assumes iteration order is monotonically increasing
    // to to ensure stableness.
    uint32_t elemCount = static_cast<uint32_t>(dInput.size());
    uint32_t blockCount = queue.DetermineGridStrideBlock(nullptr, 0u,
                                                         StaticThreadPerBlock1D(),
                                                         elemCount);

    LRCounter* dTempPtr = reinterpret_cast<LRCounter*>(dTempMemory.data());
    size_t counterAmount = (blockCount + 1);
    assert(dTempMemory.size_bytes() <= sizeof(LRCounter) * counterAmount);
    Span<LRCounter> dCounters = Span<LRCounter>(dTempPtr, counterAmount);
    //
    dCounters[0] = LRCounter{.left = 0, .right = 0};
    queue.IssueWorkLambda
    (
        "KCBinaryPartition-FindOffsets",
        DeviceWorkIssueParams{.workCount = uint32_t(dInput.size())},
        [=](KernelCallParams kp)
        {
            if(op(dInput[kp.GlobalId()]))
                dCounters[kp.blockId + 1].left += 1;
            else
                dCounters[kp.blockId + 1].right += 1;
        }
    );
    queue.IssueBlockLambda
    (
        "KCBinaryPartition-SumOffsets",
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        [=](KernelCallParams)
        {
            LRCounter offsetSum = dCounters[0];
            for(uint32_t i = 0; i < uint32_t(dCounters.size()); i++)
            {
                // Inclusive Scan
                dCounters[i + 1].left += offsetSum.left;
                dCounters[i + 1].right += offsetSum.right;
                offsetSum = dCounters[i + 1];
            }
        }
    );
    // From now on counters are write offsets
    queue.IssueWorkLambda
    (
        "KCBinaryPartition-WriteData",
        DeviceWorkIssueParams{.workCount = uint32_t(dInput.size())},
        [=](KernelCallParams kp)
        {

            MRAY_SHARED_MEMORY Vector2ui localOffset;
            if(kp.threadId == 0) localOffset = Vector2ui::Zero();

            // The first thread sets the offset counter from the dCounters variable.
            if(kp.GlobalId() == 0)
                dEndOffset[0] = dCounters.back().left;

            const auto& globalOffset = dCounters[kp.blockId];
            //
            if(op(dInput[kp.GlobalId()]))
                dOutput[globalOffset.left + (localOffset[0]++)] = dInput[kp.GlobalId()];
            else
                dOutput[globalOffset.right + (localOffset[1]++)] = dInput[kp.GlobalId()];

            if constexpr(MRAY_IS_DEBUG)
            {
                if(kp.threadId == kp.blockSize - 1)
                {
                    LRCounter nextCounter = dCounters[kp.blockId];
                    assert(localOffset[0] == nextCounter.left - globalOffset.left);
                    assert(localOffset[1] == nextCounter.right - globalOffset.right);
                }
            }
        }
    );
}
}