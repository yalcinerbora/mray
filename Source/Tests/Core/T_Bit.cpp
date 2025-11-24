#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "Core/BitFunctions.h"

#include "GTestWrappers.h"

TEST(BitTest, Compose)
{
    auto x = Bit::Compose<12, 8>(0xFFFu, 0xFFu);
    EXPECT_EQ(x, 0xFFFFFu);
}