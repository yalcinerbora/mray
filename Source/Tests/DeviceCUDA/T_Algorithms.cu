#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/TextureCUDA.h"
#include "Core/Log.h"

template <class K, class V>
struct KVType
{
    using Key = K;
    using Value = V;
};

template <class T>
class DeviceAlorithmsTest : public testing::Test
{
    public:
    using KeyType = typename T::Key;
    using ValueType = typename T::Value;
};

// Too many types dramatically increases compilation time due to "cuda::cub"
using Implementations = ::testing::Types
<
    KVType<uint32_t, uint32_t>,
    KVType<uint64_t, Float>,
    KVType<uint32_t, Vector2>
>;

TYPED_TEST_SUITE(DeviceAlorithmsTest, Implementations);

template <class T>
struct Adder
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    T operator()(const T& l, const T& r)
    {
        return r + l;
    }
};

// Temporarily define a increment for iota
template <int D, class T>
Vector<D, T> operator++(Vector<D, T>& a)
{
    for(int i = 0; i < D; i++)
        a[i] += 1;
    return a;
}

template<class Value>
void ReduceTest(const GPUSystem& system)
{
    static constexpr size_t ElementCount = 1'000;
    size_t tempMemSize = DeviceAlgorithms::ReduceTMSize<Value>(ElementCount);
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutput;
    Span<Byte> dTempMemory;
    MemAlloc::AllocateMultiData(std::tie(dInputs, dTempMemory, dOutput),
                                mem, {ElementCount, tempMemSize, 1});

    Span<Value> dOutputExact = dOutput.subspan(0, 1);
    Span<Value> dInputsExact = dInputs.subspan(0, ElementCount);

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));

    const GPUQueue& queue = system.BestDevice().GetQueue(0);
    queue.MemcpyAsync(dInputsExact, Span<const Value>(hInputs.begin(), hInputs.end()));

    DeviceAlgorithms::Reduce(Span<Value, 1>(dOutputExact), dTempMemory,
                             ToConstSpan(dInputs.subspan(0, ElementCount)),
                             Value(0), queue, Adder<Value>());

    Value hResult;
    queue.MemcpyAsync(Span<Value>(&hResult, 1), ToConstSpan(dOutputExact));
    queue.MemsetAsync(dOutputExact, 0x00);

    // Do the reduction again with a lambda
    DeviceAlgorithms::Reduce(Span<Value, 1>(dOutputExact), dTempMemory,
                             ToConstSpan(dInputs.subspan(0, ElementCount)),
                             Value(0), queue,
                             []MRAY_HYBRID(const Value& l, const Value& r)
                             {
                                 return l + r;
                             });

    Value hResultLambda;
    queue.MemcpyAsync(Span<Value>(&hResultLambda, 1), ToConstSpan(dOutputExact));
    queue.Barrier().Wait();

    Value result = (Value(ElementCount) * (Value(ElementCount) - Value(1))) / Value(2);
    if constexpr(std::is_arithmetic_v<Value>)
    {
        if constexpr(std::is_integral_v<Value>)
        {
            EXPECT_EQ(result, hResult);
            EXPECT_EQ(result, hResultLambda);
        }
        else
        {
            EXPECT_FLOAT_EQ(result, hResult);
            EXPECT_FLOAT_EQ(result, hResultLambda);
        }
    }
    else
    {
        for(int d = 0; d < Value::Dims; d++)
        {
            if constexpr(std::is_integral_v<typename Value::InnerType>)
            {
                EXPECT_EQ(result[d], hResult[d]);
                EXPECT_EQ(result[d], hResultLambda[d]);
            }
            else
            {
                EXPECT_FLOAT_EQ(result[d], hResult[d]);
                EXPECT_FLOAT_EQ(result[d], hResultLambda[d]);
            }
        }
    }
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

TYPED_TEST(DeviceAlorithmsTest, Reduce)
{
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    ReduceTest<Value>(system);
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