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
    EXPECT_THAT(result.AsArray(),
                Pointwise(FloatNear(epsilon), expected.AsArray()));
}

template<ArrayLikeC T>
static void EXPECT_EQUAL_MRAY(const T& result, const T& expected)
{
    using ::testing::Pointwise;
    using ::testing::FloatEq;
    using ::testing::Eq;
    if constexpr(std::is_floating_point_v<typename T::InnerType>)
        EXPECT_THAT(result.AsArray(),
                    Pointwise(FloatEq(), expected.AsArray()));
    else
        EXPECT_THAT(result.AsArray(),
                    Pointwise(Eq(), expected.AsArray()));
}

template<ArrayLikeC T>
requires std::floating_point<typename T::InnerType>
static void EXPECT_EQUAL_MRAY(const T& result, const T& expected,
                              typename T::InnerType epsilon)
{
    using ::testing::Pointwise;
    using ::testing::FloatNear;
    EXPECT_THAT(result.AsArray(),
                Pointwise(FloatNear(epsilon), expected.AsArray()));
}

template<ArrayLikeC T>
static void ASSERT_EQUAL_MRAY(const T& result, const T& expected)
{
    using ::testing::Pointwise;
    using ::testing::FloatEq;
    using ::testing::Eq;

    if constexpr(std::is_floating_point_v<typename T::InnerType>)
        ASSERT_THAT(result.AsArray(),
                    Pointwise(FloatEq(), expected.AsArray()));
    else
        ASSERT_THAT(result.AsArray(),
                    Pointwise(Eq(), expected.AsArray()));
}

template<ArrayLikeC T>
requires std::floating_point<typename T::InnerType>
static void ASSERT_EQUAL_MRAY(const T& result, const T& expected,
                              typename T::InnerType epsilon)
{
    using ::testing::Pointwise;
    using ::testing::FloatNear;
    ASSERT_THAT(result.AsArray(),
                Pointwise(FloatNear(epsilon), expected.AsArray()));
}