#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "T_AlgTypes.h"

// Temporarily define a increment for iota
template <int D, class T>
Vector<D, T> operator++(Vector<D, T>& a)
{
    for(int i = 0; i < D; i++)
        a[i] += 1;
    return a;
}

template<bool IsAscending, class Key, class Value>
void RadixSortTest(const GPUSystem& system)
{
    using DeviceAlgorithms::RadixSortTMSize;
    using DeviceAlgorithms::RadixSort;

    static constexpr size_t ElementCount = 1'0;
    size_t tempMemSize = RadixSortTMSize<IsAscending, Key, Value>(ElementCount);
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
    dKeys[0] = dKeys[0].subspan(0, ElementCount);
    dKeys[1] = dKeys[1].subspan(0, ElementCount);
    dValues[0] = dValues[0].subspan(0, ElementCount);
    dValues[1] = dValues[1].subspan(0, ElementCount);


    Span<Span<Key>, 2> dKeysDB = Span<Span<Key>, 2>(dKeys);
    Span<Span<Value>, 2> dValuesDB = Span<Span<Value>, 2>(dValues);

    std::vector<Key> hKeys(ElementCount, Key(0));
    std::vector<Value> hValues(ElementCount, Value(0));
    std::vector<Key> hOldKeys(ElementCount, Key(0));

    std::iota(hKeys.begin(), hKeys.end(), Key(0));
    std::shuffle(hKeys.begin(), hKeys.end(), std::mt19937(123));
    std::iota(hValues.begin(), hValues.end(), Value(0));
    hOldKeys = hKeys;

    const GPUQueue& queue = system.BestDevice().GetQueue(0);
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