#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

template <class T>
MRAY_HOST
size_t ExclusiveScanTMSize(size_t elementCount, const GPUQueueCPU& q)
{
    uint32_t blockCount = q.DetermineGridStrideBlock(nullptr, 0u, StaticThreadPerBlock1D(),
                                                     uint32_t(elementCount));
    return (blockCount + 1) * sizeof(T);
}

template <class T, class BinaryOp>
MRAY_HOST inline
void InclusiveSegmentedScan(Span<T> dScannedValues,
                            Span<const T> dValues,
                            uint32_t segmentSize,
                            const T& identityElement,
                            const GPUQueueCPU& queue,
                            BinaryOp&& op)
{
    using namespace std::string_view_literals;
    assert(dValues.size() == dScannedValues.size());
    assert(dValues.size() % segmentSize == 0);

    // Dedicate a thread for each segment
    uint32_t segmentCount = uint32_t(dValues.size() / segmentSize);
    queue.IssueBlockLambda
    (
        "KCInclusiveSegmentedScan"sv,
        DeviceBlockIssueParams
        {
            .gridSize = segmentCount,
            .blockSize = 1
        },
        [=](KernelCallParams kp)
        {
            for(uint32_t bId = kp.blockId; bId < segmentCount; bId += kp.gridSize)
            {
                auto dLocalValues = dValues.subspan(bId * segmentSize, segmentSize);
                auto dLocalOut = dScannedValues.subspan(bId * segmentSize, segmentSize);

                T sum = identityElement;
                for(uint32_t i = 0; i < segmentSize; i++)
                {
                    sum = op(sum, dLocalValues[i]);
                    dLocalOut[i] = sum;
                }
            }
        }
    );
}

template <class T, class BinaryOp>
MRAY_HOST
void ExclusiveScan(Span<T> dScannedValues,
                   Span<Byte> dTempMem,
                   Span<const T> dValues,
                   const T& initialValue,
                   const GPUQueueCPU& queue,
                   BinaryOp&& op)
{
    using namespace std::string_view_literals;
    uint32_t blockCount = queue.DetermineGridStrideBlock(nullptr, 0u, StaticThreadPerBlock1D(),
                                                         dValues.size());
    T* dLocalSumPtr = reinterpret_cast<T*>(dTempMem.data());
    assert(dTempMem.size_bytes() >= blockCount * sizeof(T));
    Span<T> dBlockSums = Span<T>(dLocalSumPtr, blockCount + 1);

    // Each thread sums partially
    queue.IssueBlockLambda
    (
       "KCExclusiveScan-ScanPartial"sv,
        DeviceBlockIssueParams{.gridSize = blockCount, .blockSize = 1u},
        [=](KernelCallParams kp)
        {
            // One thread set this value as well
            if(kp.blockId == 0) dBlockSums[0] = initialValue;
            //
            for(uint32_t bId = kp.blockId; bId < blockCount; bId += kp.gridSize)
            {
                size_t curStart = bId * StaticThreadPerBlock1D();
                size_t nextStart = std::min((bId + 1) * StaticThreadPerBlock1D(),
                                            dValues.size());
                size_t localSize = nextStart - curStart;
                if(localSize == 0) continue;

                Span<const T> dLocalValues = dValues.subspan(curStart, localSize);
                Span<const T> dLocalOut = dScannedValues.subspan(curStart, localSize);

                T localSum = dLocalValues[0];
                for(uint32_t i = 1; i < uint32_t(localSize); i++)
                {
                    dLocalOut[i] = localSum;
                    localSum = op(localSum, dLocalValues[i]);
                }
                dBlockSums[bId + 1] = localSum;
            }
        }
    );

    queue.IssueBlockLambda
    (
        "KCExclusiveScan-ScanBlockSums"sv,
        DeviceBlockIssueParams{.gridSize = 1u, .blockSize = 1u},
        [=](KernelCallParams)
        {
            for(size_t i = 1; i < dBlockSums[i].size(); i++)
                dBlockSums[i] = op(dBlockSums[i] + dBlockSums[i - 1]);
        }
    );

    queue.IssueWorkLambda
    (
        "KCExclusiveScan-AddBlockOffsets"sv,
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dValues.size())},
        [=](KernelCallParams kp)
        {
            T sum;
            for(size_t i = kp.GlobalId(); i < dBlockSums[i].size(); i += kp.TotalSize())
            {
                bool isFirstThreadInBlock = (i % StaticThreadPerBlock1D() == 0);
                if(isFirstThreadInBlock)
                    dValues[i] = dBlockSums[i];
                else
                    dValues[i] = op(dValues[i], dBlockSums[i]);
            }
        }
    );
}

}