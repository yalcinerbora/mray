#pragma once
// IWYU pragma: private; include "AlgBinaryPartition.h"

#include <new>

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

struct alignas(MRayCPUCacheLineDestructive) LRCounter
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
    // Algorithm Summary:
    //   ---- O(2N + C) where C is the core count.
    //   ---- O(C) memory (two 32-bit integers per core (should be size_t)
    //        aligned to prevent cache flushes or data persistence between threads.
    //
    //
    // For both parts, the partition is stable. Each thread
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
    uint32_t workPerBlock = Math::DivideUp(elemCount, blockCount);

    LRCounter* dTempPtr = reinterpret_cast<LRCounter*>(dTempMemory.data());
    size_t counterAmount = (blockCount + 1);
    assert(dTempMemory.size_bytes() >= sizeof(LRCounter) * counterAmount);
    Span<LRCounter> dCounters = Span<LRCounter>(dTempPtr, counterAmount);
    //
    queue.MemsetAsync(dCounters, 0x00);
    queue.IssueBlockLambda
    (
        "KCBinaryPartition-FindOffsets",
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = 1u
        },
        [=](KernelCallParams kp)
        {
            uint32_t offset = kp.blockId * workPerBlock;
            uint32_t localWCount = std::min(offset + workPerBlock, elemCount) - offset;
            Span<const T> dLocalInput = dInput.subspan(offset, localWCount);
            for(const T& v : dLocalInput)
            {
                if(op(v))   dCounters[kp.blockId + 1].left += 1;
                else        dCounters[kp.blockId + 1].right += 1;
            }
        }
    );
    queue.IssueBlockLambda
    (
        "KCBinaryPartition-SumOffsets",
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        [=](KernelCallParams)
        {
            uint32_t offset = dCounters[0].left;
            assert(offset == 0);
            // Inclusive Scan
            for(uint32_t i = 1; i < uint32_t(dCounters.size()); i++)
            {
                dCounters[i].left += offset;
                offset = dCounters[i].left;
            }
            for(uint32_t i = 0; i < uint32_t(dCounters.size()); i++)
            {
                dCounters[i].right += offset;
                offset = dCounters[i].right;
            }
            assert(offset == elemCount);
        }
    );
    // From now on counters are write offsets
    queue.IssueBlockLambda
    (
        "KCBinaryPartition-WriteData",
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = 1u
        },
        [=](KernelCallParams kp)
        {
            Vector2ui localOffset = Vector2ui::Zero();
            // The first thread sets the offset counter from the dCounters variable.
            if(kp.blockId == 0) dEndOffset[0] = dCounters.back().left;

            const auto& globalOffset = dCounters[kp.blockId];

            uint32_t offset = kp.blockId * workPerBlock;
            uint32_t localWCount = std::min(offset + workPerBlock, elemCount) - offset;
            if(offset >= dInput.size())
                MRAY_DEBUG_BREAK;
            Span<const T> dLocalInput = dInput.subspan(offset, localWCount);
            for(const T& v : dLocalInput)
            {
                if(((globalOffset.left + localOffset[0]) >= dOutput.size()) &&
                   ((globalOffset.right + localOffset[1]) >= dOutput.size()))
                {
                    MRAY_DEBUG_BREAK;
                }

                if(op(v))   dOutput[globalOffset.left +  (localOffset[0]++)] = v;
                else        dOutput[globalOffset.right + (localOffset[1]++)] = v;
            }
            if constexpr(MRAY_IS_DEBUG)
            {
                [[maybe_unused]]
                LRCounter nextCounter = dCounters[kp.blockId + 1];
                assert(localOffset[0] == nextCounter.left - globalOffset.left);
                assert(localOffset[1] == nextCounter.right - globalOffset.right);
            }
        }
    );
}
}