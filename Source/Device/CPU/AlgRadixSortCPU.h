#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

static constexpr int32_t BIT_PER_PASS = 8;
static constexpr int32_t OFFSET_BUFFER_SIZE = (1u << BIT_PER_PASS);

using CountBuffer = std::array<uint32_t, OFFSET_BUFFER_SIZE>;
using OffsetBuffer = std::array<uint32_t, OFFSET_BUFFER_SIZE + 1>;

template <bool IsAscending, class K, class V>
MRAY_GPU inline
uint32_t RadixSortKVSingleThread(Span<Span<K>, 2> dKeyDoubleBuffer,
                                 Span<Span<V>, 2> dValueDoubleBuffer,
                                 Vector2ui bitRange)
{
    OffsetBuffer offsetBuffer;
    CountBuffer countBuffer;
    std::fill(countBuffer.begin(), countBuffer.end(), 0u);

    using UIntType = std::conditional_t<(sizeof(K) > sizeof(uint32_t)), uint64_t, uint32_t>;
    uint32_t inputBufferIndex = 0;
    uint32_t outputBufferIndex = 1;

    // Sort MSB to LSB
    int32_t totalBits = int32_t(bitRange[1] - bitRange[0]);
    for(int32_t pass = 0; pass < totalBits; pass += BIT_PER_PASS)
    {
        std::array<UIntType, 2> curBitRange =
        {
            UIntType(std::max(int32_t(bitRange[1]) - (pass + 0) * BIT_PER_PASS, 0)),
            UIntType(std::max(int32_t(bitRange[1]) - (pass + 1) * BIT_PER_PASS, 0))
        };
        uint32_t curPassBitCount = uint32_t(curBitRange[1] - curBitRange[0]);

        Span<K> dRKeyBuffer = dKeyDoubleBuffer[inputBufferIndex];
        Span<V> dRValBuffer = dValueDoubleBuffer[inputBufferIndex];
        //
        Span<K> dWKeyBuffer = dKeyDoubleBuffer[outputBufferIndex];
        Span<V> dWValBuffer = dValueDoubleBuffer[outputBufferIndex];

        for(const K& key : dRKeyBuffer)
        {
            UIntType keyBits = 0u;
            std::memcpy(&keyBits, &key, sizeof(K));
            uint32_t bucketIndex = uint32_t(Bit::FetchSubPortion(keyBits, curBitRange));
            countBuffer[bucketIndex + 1] += 1;
        }

        std::inclusive_scan(countBuffer.cbegin(),
                            countBuffer.cbegin() + curPassBitCount,
                            offsetBuffer.begin() + 1);
        offsetBuffer[0] = 0u;

        for(uint32_t i = 0; i < uint32_t(dRKeyBuffer.size()); i++)
        {
            K& key = dRKeyBuffer[i];
            V& val = dRValBuffer[i];

            UIntType keyBits = 0u;
            std::memcpy(&keyBits, &key, sizeof(K));
            uint32_t bucketIndex = uint32_t(Bit::FetchSubPortion(keyBits, curBitRange));

            uint32_t bucketSize = offsetBuffer[bucketIndex + 1] - offsetBuffer[bucketIndex];
            uint32_t localIndex = bucketSize - countBuffer[bucketIndex]--;
            uint32_t pushLoc = offsetBuffer[bucketIndex] + localIndex;

            dWKeyBuffer[pushLoc] = std::move(key);
            dWValBuffer[pushLoc] = std::move(val);
        }

        if constexpr(MRAY_IS_DEBUG)
        {
            assert(std::all_of(offsetBuffer.cbegin(), offsetBuffer.cend(),
                               [](uint32_t v){return v == 0; }));
        }
        std::swap(inputBufferIndex, outputBufferIndex);
    }
    return outputBufferIndex;
}


template <bool IsAscending, class K, class V>
MRAY_HOST inline
size_t SegmentedRadixSortTMSize(size_t,
                                size_t,
                                const GPUQueueCPU&)
{
    return 0u;
}

template <bool IsAscending, class K, class V>
MRAY_HOST inline
size_t RadixSortTMSize(size_t elementCount,
                       const GPUQueueCPU& q)
{
    uint32_t blockCount = q.DetermineGridStrideBlock(nullptr, 0u, StaticThreadPerBlock1D(),
                                                     uint32_t(elementCount));
    return sizeof(CountBuffer) + sizeof(OffsetBuffer) * blockCount;
}

template <bool IsAscending, class K, class V>
MRAY_HOST inline
uint32_t RadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                   Span<Span<V>, 2> dValueDoubleBuffer,
                   Span<Byte> dTempMemory,
                   const GPUQueueCPU& queue,
                   const Vector2ui& bitRange)
{
    using namespace std::string_view_literals;

    uint32_t blockCount = queue.DetermineGridStrideBlock(nullptr, 0u, StaticThreadPerBlock1D(),
                                                         uint32_t(dValueDoubleBuffer[0].size()));
    Byte* dOffsetBufPtr = dTempMemory.data();
    Byte* dCountBufPtr = dTempMemory.data() + sizeof(OffsetBuffer) * blockCount;
    static_assert((sizeof(OffsetBuffer)) % alignof(CountBuffer) == 0, "Alignment mismatch!");
    assert(dTempMemory.size_bytes() <= (sizeof(CountBuffer) + sizeof(OffsetBuffer)) * blockCount);

    // TODO: Set these buffers from temp memory
    Span<OffsetBuffer> dOffsetBuffers(reinterpret_cast<OffsetBuffer*>(dOffsetBufPtr), blockCount);
    Span<CountBuffer> dCountBuffers(reinterpret_cast<CountBuffer*>(dCountBufPtr), blockCount);

    // Initially set the count buffers to zer
    queue.MemsetAsync(dCountBuffers, 0x00);

    using UIntType = std::conditional_t<(sizeof(K) > sizeof(uint32_t)), uint64_t, uint32_t>;
    uint32_t inputBufferIndex = 0;
    uint32_t outputBufferIndex = 1;
    // Sort MSB to LSB
    int32_t totalBits = int32_t(bitRange[1] - bitRange[0]);
    for(int32_t pass = 0; pass < totalBits; pass += BIT_PER_PASS)
    {
        std::array<UIntType, 2> curBitRange =
        {
            UIntType(std::max(int32_t(bitRange[1]) - (pass + 0) * BIT_PER_PASS, 0)),
            UIntType(std::max(int32_t(bitRange[1]) - (pass + 1) * BIT_PER_PASS, 0))
        };
        uint32_t curPassBitCount = uint32_t(curBitRange[1] - curBitRange[0]);

        Span<K> dRKeyBuffer = dKeyDoubleBuffer[inputBufferIndex];
        Span<V> dRValBuffer = dValueDoubleBuffer[inputBufferIndex];
        //
        Span<K> dWKeyBuffer = dKeyDoubleBuffer[outputBufferIndex];
        Span<V> dWValBuffer = dValueDoubleBuffer[outputBufferIndex];

        queue.IssueWorkLambda
        (
            "KCRadixSort-DetermineBuckets"sv,
            DeviceWorkIssueParams{.workCount = uint32_t(dRKeyBuffer.size())},
            [=](KernelCallParams kp)
            {
                UIntType keyBits = 0u;
                std::memcpy(&keyBits, &dRKeyBuffer[kp.GlobalId()], sizeof(K));
                uint32_t bucketIndex = uint32_t(Bit::FetchSubPortion(keyBits, curBitRange));
                dCountBuffers[kp.blockId][bucketIndex + 1] += 1;
            }
        );
        //
        queue.IssueBlockLambda
        (
            "KCRadixSort-ScanBuckets"sv,
            DeviceBlockIssueParams{.gridSize = 1u, .blockSize = 1u},
            [=](KernelCallParams)
            {
                for(uint32_t bId = 0; bId < blockCount; bId++)
                {
                    dCountBuffers[bId][0] = (bId != 0) ? dCountBuffers[bId - 1][0] : 0;
                    for(uint32_t i = 1; i < (curPassBitCount + 1); i++)
                    {
                        dOffsetBuffers[bId][i] = (dOffsetBuffers[bId][i - 1] +
                                                  dCountBuffers[bId][i]);
                    }
                }
            }
        );
        //
        queue.IssueWorkLambda
        (
            "KCRadixSort-DetermineBuckets"sv,
            DeviceWorkIssueParams{.workCount = uint32_t(dRKeyBuffer.size())},
            [=](KernelCallParams kp)
            {
                K& key = dRKeyBuffer[kp.GlobalId()];
                V& val = dRValBuffer[kp.GlobalId()];
                const auto offsetBuffer = dOffsetBuffers[kp.blockId];
                auto& countBuffer = dCountBuffers[kp.blockId];

                UIntType keyBits = 0u;
                std::memcpy(&keyBits, &key, sizeof(K));
                uint32_t bucketIndex = uint32_t(Bit::FetchSubPortion(keyBits, curBitRange));

                uint32_t bucketSize = offsetBuffer[bucketIndex + 1] - offsetBuffer[bucketIndex];
                uint32_t localIndex = bucketSize - countBuffer[bucketIndex]--;
                uint32_t pushLoc = offsetBuffer[bucketIndex] + localIndex;

                dWKeyBuffer[pushLoc] = std::move(key);
                dWValBuffer[pushLoc] = std::move(val);
            }
        );

        if constexpr(MRAY_IS_DEBUG)
        {
            queue.IssueBlockLambda
            (
                "KCRadixSort-ScanBuckets"sv,
                DeviceBlockIssueParams{.gridSize = 1u, .blockSize = 1u},
                [=](KernelCallParams)
                {
                    [[maybe_unused]]
                    bool allZero = true;

                    for(uint32_t bId = 0; bId < blockCount; bId++)
                    for(uint32_t i = 0; i < BIT_PER_PASS; i++)
                    {
                        allZero &= (dCountBuffers[bId][i] == 0);
                    }
                    assert(allZero);
                }
            );
        }
        std::swap(inputBufferIndex, outputBufferIndex);
    }
    return outputBufferIndex;
}

template <bool IsAscending, class K, class V>
MRAY_HOST inline
uint32_t SegmentedRadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                            Span<Span<V>, 2> dValueDoubleBuffer,
                            Span<Byte>,
                            Span<const uint32_t> dSegmentRanges,
                            const GPUQueueCPU& queue,
                            const Vector2ui& bitRange)
{
    using namespace std::string_view_literals;

    // Cheat a little here:
    // Dedicate a block to each segment,
    // use std::stable_sort.
    // TODO: std::stable_sort temp allocates,
    // we have extra buffer, if only we could give it to
    // the function... So rewrite this when this will be a perf
    // issue.
    uint32_t segmentCount = uint32_t(dSegmentRanges.size() - 1u);
    uint32_t passCount = Math::DivideUp(bitRange[1] - bitRange[0], uint32_t(BIT_PER_PASS));
    uint32_t expectedOutIndex = ((passCount & 0x1) == 0) ? 0 : 1;
    queue.IssueBlockLambda
    (
        "KCSegmentedRadixSort"sv,
        DeviceBlockIssueParams{.gridSize = segmentCount, .blockSize = 1u},
        [=](KernelCallParams kp)
        {
            for(uint32_t bId = kp.blockId; bId < segmentCount; bId += kp.gridSize)
            {
                Vector2ui sRange = Vector2ui(dSegmentRanges[bId + 0],
                                             dSegmentRanges[bId + 1]);
                uint32_t sSize = sRange[1] - sRange[0];
                std::array<Span<K>, 2> dBlockKeys =
                {
                    dKeyDoubleBuffer[0].subspan(sRange[0], sSize),
                    dKeyDoubleBuffer[1].subspan(sRange[0], sSize)
                };
                std::array<Span<V>, 2> dBlockValues =
                {
                    dValueDoubleBuffer[0].subspan(sRange[0], sSize),
                    dValueDoubleBuffer[1].subspan(sRange[0], sSize)
                };

                [[maybe_unused]]
                uint32_t localOutIndex = RadixSortKVSingleThread<IsAscending>
                (
                    Span<Span<K>, 2>(dBlockKeys),
                    Span<Span<V>, 2>(dBlockValues),
                    bitRange
                );
                assert(localOutIndex == expectedOutIndex);
            }
        }
    );
    // We do stuff in-place, so first buffer has the value
    return expectedOutIndex;
}

}