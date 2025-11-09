
#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "GTestWrappers.h"

#include "Tracer/HashGrid.h"

#include "Core/Math.h"

TEST(HashGrid, Deterministic)
{
    static constexpr uint32_t SEED = 0;
    static constexpr uint32_t SAMPLE_COUNT = 10'000;


    Span<SpatioDirCode> dHashes;
    Span<uint32_t, 1> dAllocCounter(nullptr, 1);

    Float degrees = Float (0.5);
    Float radiansHalf = MathConstants::DegToRadCoef<Float>() * degrees * Float(0.5);

    HashGridView hgView
    {
        .dHashes = dHashes,
        .dAllocCounter = dAllocCounter,
        .camLocation = Vector3(1, 2, 3),
        .hashGridRegion = AABB3(Vector3(-1, -2, -3), Vector3(3, 1, 2)),
        .baseRegionDelta = Float(0.5),
        .baseRegionDim = 16384,
        .normalDelta = 2,
        .normalRegionDim = 2,
        .maxLevelOffset = 3,
        .maxEntryLimit = 1'000,
        .tanConeHalfTimes2 = Math::Tan(radiansHalf) * Float(2)
    };

    std::vector<SpatioDirCode> a, b;
    {
        BackupRNGState s = BackupRNG::GenerateState(SEED);
        BackupRNG rng(s);
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
            a.push_back(hgView.GenCodeStochastic(Vector3(rng.NextFloat(),
                                                         rng.NextFloat(),
                                                         rng.NextFloat()),
                                                 Vector3(rng.NextFloat(),
                                                         rng.NextFloat(),
                                                         rng.NextFloat()),
                                                 rng));
    }
    {
        BackupRNGState s = BackupRNG::GenerateState(SEED);
        BackupRNG rng(s);
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
            b.push_back(hgView.GenCodeStochastic(Vector3(rng.NextFloat(),
                                                         rng.NextFloat(),
                                                         rng.NextFloat()),
                                                 Vector3(rng.NextFloat(),
                                                         rng.NextFloat(),
                                                         rng.NextFloat()),
                                                 rng));
    }

    for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
    {
        EXPECT_EQ(b[i], a[i]);
    }


}

