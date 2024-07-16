#include <gtest/gtest.h>
#include <random>

#include "Tracer/Filters.h"

#include "Device/GPUSystem.hpp"

TEST(Filter_Box, ZeroVariance)
{
    //GaussianFilter f(r);
}

TEST(Filter_Tent, ZeroVariance)
{

}

TEST(Filter_Gaussian, ZeroVariance)
{

}

TEST(Filter_MitchellNetravali, ZeroVariance)
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
        Float r, b;
        if(f == 0)
            //r = MathConstants::Epsilon<Float>();
            r = 2;
        else
            r = distR(rng);

        const Float integral = 1;
        MitchellNetravaliFilter filter(r);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Vector2 xi;
/*            if(i == 0) xi = Vector2(0);
            else if(i == 1) xi = Vector2(MathFunctions::PrevFloat<Float>(1));
            else */xi = Vector2(dist01(rng), dist01(rng));

            using namespace Distribution;
            auto result = filter.Sample(xi);
            // Check pdf from the function
            Float pdfFromFunc = filter.Pdf(result.value);
            EXPECT_NEAR(pdfFromFunc, result.pdf, VeryLargeEpsilon<Float>());

            // Evaluate the function
            Float eval = filter.Evaluate(result.value);
            Float estimate = eval / result.pdf;
            // Since this is zero variance estimate,
            // the estimate should exactly match
            // actual integral.
            // TODO: This is somewhat bad we can only get
            // 10^-3 level of precision? (Is something wrong?)
            EXPECT_NEAR(integral, estimate, VeryLargeEpsilon<Float>());
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        EXPECT_NEAR(integral, total, LargeEpsilon<Float>());
    }
}