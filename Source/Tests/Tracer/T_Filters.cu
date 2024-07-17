#include <gtest/gtest.h>
#include <random>

#include "Tracer/Filters.h"

#include "Device/GPUSystem.hpp"

template<class Filter>
void TestFilter(bool checkZeroVariance)
{
    using namespace MathConstants;
    static constexpr uint32_t SAMPLE_COUNT = 128;
    static constexpr uint32_t FUNCTION_COUNT = 16;
    // Function overall min/max
    static constexpr Float RADIUS_MAX = 16;

    std::mt19937 rng(332);
    using UniformDist = std::uniform_real_distribution<Float>;

    UniformDist dist01;
    UniformDist distR(0, RADIUS_MAX);
    for(uint32_t f = 0; f < FUNCTION_COUNT; f++)
    {
        Float r;
        if(f == 0)  r = MathConstants::HugeEpsilon<Float>();
        else        r = distR(rng);
        //
        const Float integral = 1;
        Filter filter(r);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Vector2 xi;
            if(i == 0) xi = Vector2(0);
            else if(i == 1) xi = Vector2(MathFunctions::PrevFloat<Float>(1));
            else xi = Vector2(dist01(rng), dist01(rng));
            // Sample
            auto result = filter.Sample(xi);

            // Check pdf from the function
            Float pdfFromFunc = filter.Pdf(result.value);
            EXPECT_NEAR(pdfFromFunc, result.pdf, HugeEpsilon<Float>());

            // Evaluate the function
            Float eval = filter.Evaluate(Vector2(result.value));
            Float estimate = eval / result.pdf;

            // Some filters (Mitchell-Netravali)
            // do not have perfect sampler;
            // thus, a single sample will not give the exact integral.
            // Skip zero variance check for these functions.
            if(checkZeroVariance)
            {
                // Since this is zero variance estimate,
                // the estimate should exactly match
                // actual integral.
                // TODO: This is somewhat bad we can only get
                // 10^-3 level of precision? (Is something wrong?)
                EXPECT_NEAR(integral, estimate,
                            VeryLargeEpsilon<Float>());
            }
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        if(!checkZeroVariance)
        {
            // For imperfect (non-zero variance) samplers,
            // %15 error is OK for 128 sample(?)
            EXPECT_NEAR(integral, total, Float(0.15));
        }
        else
        {
            // For the rest, do an aggressive check
            EXPECT_NEAR(integral, total, LargeEpsilon<Float>());
        }
    }
}

TEST(Filter_Box, ZeroVariance)
{
    TestFilter<BoxFilter>(true);
}

TEST(Filter_Tent, ZeroVariance)
{
    TestFilter<TentFilter>(true);
}

TEST(Filter_Gaussian, ZeroVariance)
{
    TestFilter<GaussianFilter>(true);
}

TEST(Filter_MitchellNetravali, ZeroVariance)
{
    TestFilter<MitchellNetravaliFilter>(false);
}