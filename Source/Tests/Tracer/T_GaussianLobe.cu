
#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "GTestWrappers.h"

#include "Tracer/DistributionFunctions.h"

#include "Core/Math.h"

TEST(GaussLobe, Sample)
{
    using GaussLobe = Distribution::GaussianLobe;
    static constexpr uint32_t SEED = 0;
    static constexpr uint32_t SAMPLE_COUNT = 10'000;
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<Float> float01;

    // TODO: These tests are not good find a way to properly test
    // later
    {
        Vector3 dir = Vector3::YAxis();
        Float kappa = Float(0);

        GaussLobe lobe(dir, kappa);
        Vector3 meanDir = Vector3::Zero();
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            Vector2 xi(float01(rng), float01(rng));
            auto s = lobe.Sample(xi);
            meanDir += s.value;

            EXPECT_NEAR(s.pdf, MathConstants::Inv4Pi<Float>(),
                        MathConstants::SmallEpsilon<Float>());
        }
        meanDir /= Float(SAMPLE_COUNT);
        EXPECT_NEAR_MRAY(meanDir, Vector3::Zero(),
                         MathConstants::HugeEpsilon<Float>());
    }
    {
        Vector3 dir = Vector3::YAxis();
        Float kappa = Float(10'000);
        GaussLobe lobe(dir, kappa);
        Vector3 meanDir = Vector3::Zero();
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            Vector2 xi(float01(rng), float01(rng));
            auto s = lobe.Sample(xi);
            meanDir += s.value;
        }
        meanDir /= Float(SAMPLE_COUNT);
        EXPECT_NEAR_MRAY(meanDir, Vector3::YAxis(),
                         MathConstants::VeryLargeEpsilon<Float>());
    }
}

