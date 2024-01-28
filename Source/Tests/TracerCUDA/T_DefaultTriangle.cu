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

    EXPECT_EQ(std::get<0>(list[0]),"Position"sv);
    EXPECT_EQ(std::get<1>(list[0]), AttributeOptionality::MR_MANDATORY);
    EXPECT_EQ(std::get<2>(list[0]).name, MRayDataEnum::MR_VECTOR_3);

    EXPECT_EQ(std::get<0>(list[1]), "Normal"sv);
    EXPECT_EQ(std::get<1>(list[1]), AttributeOptionality::MR_OPTIONAL);
    EXPECT_EQ(std::get<2>(list[1]).name, MRayDataEnum::MR_QUATERNION);

    EXPECT_EQ(std::get<0>(list[2]), "UV0"sv);
    EXPECT_EQ(std::get<1>(list[2]), AttributeOptionality::MR_OPTIONAL);
    EXPECT_EQ(std::get<2>(list[2]).name, MRayDataEnum::MR_VECTOR_2);

    EXPECT_EQ(std::get<0>(list[3]), "Index"sv);
    EXPECT_EQ(std::get<1>(list[3]), AttributeOptionality::MR_MANDATORY);
    EXPECT_EQ(std::get<2>(list[3]).name, MRayDataEnum::MR_VECTOR_3UI);
}



TEST(DefaultPrimitives, Triangle_Load)
{
    //GPUSystem s;

    //std::vector<PrimCount> primCounts;
    //primCounts.push_back(PrimCount{.primCount = 1, .attributeCount = 3});

    //PrimGroupTriangle triGroup(0, s);
    //PrimBatchKey batch = triGroup.Reserve(primCounts);
    //triGroup.CommitReservations();

    //PrimAttributeInfoList list = triGroup.AttributeInfo();

    //std::vector<Byte> a;

    //triGroup.PushAttribute(batch, 0,
    //                       )




}
