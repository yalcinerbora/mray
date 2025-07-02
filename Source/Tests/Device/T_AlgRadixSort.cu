#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgForward.h"
#include "Device/GPUAlgRadixSort.h"

#include "T_AlgTypes.h"

template<bool IsAscending, class Key, class Value>
void RadixSortTest(const GPUSystem& system)
{
    using DeviceAlgorithms::RadixSortTMSize;
    using DeviceAlgorithms::RadixSort;

    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    static constexpr size_t ElementCount = 10;
    size_t tempMemSize = RadixSortTMSize<IsAscending, Key, Value>(ElementCount, queue);
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Key> dKeys[2];
    Span<Value> dValues[2];
    Span<Byte> dTempMemory;
    MemAlloc::AllocateMultiData(std::tie(dKeys[0], dKeys[1],
                                         dValues[0], dValues[1],
                                         dTempMemory),
                                mem,
                                {ElementCount, ElementCount,
                                 ElementCount, ElementCount,
                                 tempMemSize});

    Span<Span<Key>, 2> dKeysDB = Span<Span<Key>, 2>(dKeys);
    Span<Span<Value>, 2> dValuesDB = Span<Span<Value>, 2>(dValues);

    std::vector<Key> hKeys(ElementCount, Key(0));
    std::vector<Value> hValues(ElementCount, Value(0));
    std::vector<Key> hOldKeys(ElementCount, Key(0));

    std::iota(hKeys.begin(), hKeys.end(), Key(0));
    std::shuffle(hKeys.begin(), hKeys.end(), std::mt19937(123));
    std::iota(hValues.begin(), hValues.end(), Value(0));
    hOldKeys = hKeys;

    queue.MemcpyAsync(dKeys[0], Span<const Key>(hKeys.begin(), hKeys.end()));
    queue.MemcpyAsync(dValues[0], Span<const Value>(hValues.begin(), hValues.end()));

    uint32_t bufferIndex = RadixSort<IsAscending>(dKeysDB, dValuesDB, dTempMemory, queue);

    // Read back
    queue.MemcpyAsync(Span<Key>(hKeys.begin(), hKeys.end()),
                      ToConstSpan(dKeysDB[bufferIndex]));
    queue.MemcpyAsync(Span<Value>(hValues.begin(), hValues.end()),
                      ToConstSpan(dValuesDB[bufferIndex]));
    queue.Barrier().Wait();

    for(size_t i = 0; i < hKeys.size(); i++)
    {
        if constexpr(IsAscending)
        {
            EXPECT_EQ(Key(i), hKeys[i]);
            EXPECT_EQ(Value(i), hValues[hOldKeys[i]]);
        }
        else
        {
            EXPECT_EQ(Key(i), ElementCount - hKeys[i] - 1);
            EXPECT_EQ(Value(i), hValues[ElementCount - hOldKeys[i] - 1]);
        }
    }
}

template<bool IsAscending, class Key, class Value>
void SegmentedRadixSortTest(const GPUSystem& system)
{
    using DeviceAlgorithms::SegmentedRadixSortTMSize;
    using DeviceAlgorithms::SegmentedRadixSort;

    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    static constexpr size_t ElementCount = 64;
    static constexpr size_t SegmentCount = 5;
    size_t tempMemSize = SegmentedRadixSortTMSize<IsAscending, Key, Value>(ElementCount,
                                                                           SegmentCount,
                                                                           queue);
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Key> dKeys[2];
    Span<Value> dValues[2];
    Span<Byte> dTempMemory;
    Span<uint32_t> dSegmentRanges;
    MemAlloc::AllocateMultiData(std::tie(dKeys[0], dKeys[1],
                                         dValues[0], dValues[1],
                                         dSegmentRanges, dTempMemory),
                                mem,
                                {ElementCount, ElementCount,
                                 ElementCount, ElementCount,
                                 SegmentCount + 1, tempMemSize});

    Span<Span<Key>, 2> dKeysDB = Span<Span<Key>, 2>(dKeys);
    Span<Span<Value>, 2> dValuesDB = Span<Span<Value>, 2>(dValues);

    std::vector<Key> hKeys(ElementCount, Key(0));
    std::vector<Value> hValues(ElementCount, Value(0));
    std::vector<Key> hOldKeys(ElementCount, Key(0));
    std::vector<uint32_t> hSegmentRanges(SegmentCount + 1, uint32_t(0));

    // Initialize segment ranges by hand
    static_assert(SegmentCount == 5, "SegmentCount is changed!");
    {
        hSegmentRanges[0] = 0;
        hSegmentRanges[1] = 21;
        hSegmentRanges[2] = 45;
        hSegmentRanges[3] = 49;
        hSegmentRanges[4] = 52;
        hSegmentRanges[5] = ElementCount;
    }

    std::iota(hKeys.begin(), hKeys.end(), Key(0));
    std::iota(hValues.begin(), hValues.end(), Value(0));
    // Shuffle locally, so that we can globally check it is become
    // iota again
    std::mt19937 rng(123);
    for(uint32_t i = 0; i < SegmentCount; i++)
    {
        std::shuffle(hKeys.begin() + hSegmentRanges[i],
                     hKeys.begin() + hSegmentRanges[i + 1], rng);
    }
    hOldKeys = hKeys;

    queue.MemcpyAsync(dKeys[0], Span<const Key>(hKeys.begin(), hKeys.end()));
    queue.MemcpyAsync(dValues[0], Span<const Value>(hValues.begin(), hValues.end()));
    queue.MemcpyAsync(dSegmentRanges, Span<const uint32_t>(hSegmentRanges.begin(),
                                                           hSegmentRanges.end()));
    uint32_t bufferIndex = SegmentedRadixSort<IsAscending>(dKeysDB, dValuesDB, dTempMemory,
                                                           dSegmentRanges, queue);
    // Read back
    queue.MemcpyAsync(Span<Key>(hKeys.begin(), hKeys.end()),
                      ToConstSpan(dKeysDB[bufferIndex]));
    queue.MemcpyAsync(Span<Value>(hValues.begin(), hValues.end()),
                      ToConstSpan(dValuesDB[bufferIndex]));
    queue.Barrier().Wait();

    for(uint32_t sId = 0; sId < SegmentCount; sId++)
    for(size_t i = hSegmentRanges[sId]; i < hSegmentRanges[sId + 1]; i++)
    {
        if constexpr(IsAscending)
        {
            EXPECT_EQ(Key(i), hKeys[i]);
            EXPECT_EQ(Value(i), hValues[hOldKeys[i]]);
        }
        else
        {
            EXPECT_EQ(Key(i), ElementCount - hKeys[i] - 1);
            EXPECT_EQ(Value(i), hValues[ElementCount - hOldKeys[i] - 1]);
        }
    }
}

TYPED_TEST(DeviceAlorithmsTest, RadixSortAscending)
{
    using Key = typename DeviceAlorithmsTest<TypeParam>::KeyType;
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;

    RadixSortTest<true, Key, Value>(system);
}

TYPED_TEST(DeviceAlorithmsTest, RadixSortDescending)
{
    using Key = typename DeviceAlorithmsTest<TypeParam>::KeyType;
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;

    RadixSortTest<false, Key, Value>(system);
}

TYPED_TEST(DeviceAlorithmsTest, SegmentedRadixSortAscending)
{
    using Key = typename DeviceAlorithmsTest<TypeParam>::KeyType;
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;

    SegmentedRadixSortTest<true, Key, Value>(system);
}

TYPED_TEST(DeviceAlorithmsTest, SegmentedRadixSortDescending)
{
    using Key = typename DeviceAlorithmsTest<TypeParam>::KeyType;
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;

    SegmentedRadixSortTest<false, Key, Value>(system);
}