#pragma once

#include "Core/MathConstants.h"
#include "Core/MathForward.h"

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

template<ArrayLikeC T>
requires(std::is_floating_point_v<typename T::InnerType>)
static void EXPECT_NEAR_MRAY(const T& result, const T& expected,
                             typename T::InnerType epsilon = MathConstants::SmallEpsilon<Float>())
{
    using ::testing::Pointwise;
    using ::testing::FloatNear;
    using ::testing::Eq;
    EXPECT_THAT(result.AsSpan(),
                Pointwise(FloatNear(epsilon), expected.AsSpan()));
}

template<ArrayLikeC T>
static void EXPECT_EQUAL_MRAY(const T& result, const T& expected)
{
    using ::testing::Pointwise;
    using ::testing::FloatEq;
    using ::testing::Eq;
    if constexpr(std::is_floating_point_v<typename T::InnerType>)
        EXPECT_THAT(result.AsSpan(),
                    Pointwise(FloatEq(), expected.AsSpan()));
    else
        EXPECT_THAT(result.AsSpan(),
                    Pointwise(Eq(), expected.AsSpan()));
}

template<ArrayLikeC T>
requires std::floating_point<typename T::InnerType>
static void EXPECT_EQUAL_MRAY(const T& result, const T& expected,
                              typename T::InnerType epsilon)
{
    using ::testing::Pointwise;
    using ::testing::FloatNear;
    EXPECT_THAT(result.AsSpan(),
                Pointwise(FloatNear(epsilon), expected.AsSpan()));
}

template<ArrayLikeC T>
static void ASSERT_EQUAL_MRAY(const T& result, const T& expected)
{
    using ::testing::Pointwise;
    using ::testing::FloatEq;
    using ::testing::Eq;

    if constexpr(std::is_floating_point_v<typename T::InnerType>)
        ASSERT_THAT(result.AsSpan(),
                    Pointwise(FloatEq(), expected.AsSpan()));
    else
        ASSERT_THAT(result.AsSpan(),
                    Pointwise(Eq(), expected.AsSpan()));
}

template<ArrayLikeC T>
requires std::floating_point<typename T::InnerType>
static void ASSERT_EQUAL_MRAY(const T& result, const T& expected,
                              typename T::InnerType epsilon)
{
    using ::testing::Pointwise;
    using ::testing::FloatNear;
    ASSERT_THAT(result.AsSpan(),
                Pointwise(FloatNear(epsilon), expected.AsSpan()));
}