#pragma once

#include <gtest/gtest.h>

#include "Core/Vector.h"

template <class K, class V>
struct KVType
{
    using Key = K;
    using Value = V;
};

template <class T>
class DeviceAlgorithmsTest : public testing::Test
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

TYPED_TEST_SUITE(DeviceAlgorithmsTest, Implementations);

// Temporarily define a increment for iota
template <unsigned int D, class T>
Vector<D, T>& operator++(Vector<D, T>& a)
{
    for(unsigned int i = 0; i < D; i++)
        a[i] += 1;
    return a;
}

// Helpers
template <class T>
void ExpectEqualVecOrArithmetic(const T& expected, const T& checked)
{
    if constexpr(std::is_arithmetic_v<T>)
    {
        if constexpr(std::is_integral_v<T>)
            EXPECT_EQ(expected, checked);
        else
            EXPECT_FLOAT_EQ(expected, checked);
    }
    else
    {
        for(unsigned int d = 0; d < T::Dims; d++)
        {
            if constexpr(std::is_integral_v<typename T::InnerType>)
                EXPECT_EQ(expected[d], checked[d]);
            else
                EXPECT_FLOAT_EQ(expected[d], checked[d]);
        }
    }
}
