#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "Core/DataStructures.h"

// TODO: Add Tests...

TEST(StaticVectorTest, Constexpr)
{
    // Only this should make sense in constexpr.
    static constexpr StaticVector<Vector3, 4> defaultTest;
}

TEST(StaticVectorTest, Constructors)
{
    StaticVector<Vector3, 4> defaultTest;
    EXPECT_DEBUG_DEATH(defaultTest[0], ".*");
}