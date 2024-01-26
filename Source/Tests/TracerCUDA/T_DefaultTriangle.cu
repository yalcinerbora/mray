#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Device/GPUSystem.h"

#include "Tracer/MaterialsDefault.h"

#include "Tracer/Lights.h"

TEST(DefaultTriangle, Basic)
{
    GPUSystem s;

    PrimGroupTriangle a(0, s);

    a.ReservePrimitiveBatch(PrimCount{.primCount = 1, .attributeCount = 3});
}
