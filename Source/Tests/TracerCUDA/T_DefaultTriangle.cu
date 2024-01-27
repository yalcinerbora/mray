#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Device/GPUSystem.h"

using namespace std::literals;

MRAY_KERNEL void KCAdd()
{

}
TEST(DefaultPrimitives, Triangle_Typecheck)
{
    GPUSystem s;

    PrimGroupTriangle triGroup(0, s);
    PrimAttributeInfoList list = triGroup.AttributeInfo();

    EXPECT_EQ(list[0].first, "Position"sv);
    EXPECT_EQ(list[0].second.name, MRayDataEnum::MR_VECTOR_3);
    EXPECT_EQ(list[1].first, "Normal"sv);
    EXPECT_EQ(list[1].second.name, MRayDataEnum::MR_QUATERNION);
    EXPECT_EQ(list[2].first, "UV0"sv);
    EXPECT_EQ(list[2].second.name, MRayDataEnum::MR_QUATERNION);
    EXPECT_EQ(list[3].first, "Index"sv);
    EXPECT_EQ(list[3].second.name, MRayDataEnum::MR_VECTOR_3UI);
}



TEST(DefaultPrimitives, Triangle_Load)
{
    GPUSystem s;

    PrimGroupTriangle triGroup(0, s);
    PrimBatchId batch = triGroup.ReservePrimitiveBatch(PrimCount{.primCount = 1, .attributeCount = 3});
    triGroup.CommitReservations();

    PrimAttributeInfoList list = triGroup.AttributeInfo();

    std::vector<Byte> a;

    //triGroup.PushAttribute(batch, 0,
    //                       )




}
