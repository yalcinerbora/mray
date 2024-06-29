#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>
#include <algorithm>
#include <numeric>

#include "Tracer/Distributions.h"
#include "Tracer/DistributionFunctions.h"

#include "GTestWrappers.h"

#include "Device/GPUSystem.hpp"

// Put it as a template for future tests (PwL maybe?)
template<class Dist2D>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSampleDist(Span<SampleT<Vector2>> dOutSamples,
                  Span<const Vector2> dRandomNumbers,
                  Span<const Dist2D, 1> dDist,
                  uint32_t sampleCount)
{
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        SampleT<Vector2> sample = dDist[0].SampleIndex(dRandomNumbers[i]);
        dOutSamples[i] = sample;
    }
}

struct DistTester2D
{
    using Dist2D = typename DistributionGroupPwC2D::Distribution;
    using DistData = typename DistributionGroupPwC2D::DistDataConst;

    GPUSystem system;
    DistributionGroupPwC2D distGroup;

    DistTester2D() : distGroup(system) {}

    std::vector<SampleT<Vector2>>
    GenSamplesIndex(const std::vector<Float>& hFunction,
                    const Vector2ui& size,
                    const std::vector<Vector2>& hRandomNumbers)
    {
        using namespace std::literals;
        const GPUQueue& queue = system.BestDevice().GetQueue(0);

        uint32_t sampleCount = static_cast<uint32_t>(hRandomNumbers.size());
        uint32_t sizeLinear = size.Multiply();

        Span<SampleT<Vector2>> dOutSamples;
        Span<Vector2> dRandomNumbers;
        Span<Float> dFunction;
        DeviceMemory mem({&system.BestDevice()}, 32_MiB, 128_MiB);
        MemAlloc::AllocateMultiData(std::tie(dFunction, dRandomNumbers,
                                             dOutSamples),
                                    mem,
                                    {sizeLinear, sampleCount, sampleCount});
        queue.MemcpyAsync(dFunction, Span<const Float>(hFunction.cbegin(), hFunction.cend()));

        uint32_t id = distGroup.Reserve(size);
        distGroup.Commit();
        distGroup.Construct(id, dFunction);

        Span<const Dist2D> dists = distGroup.DeviceDistributions();
        EXPECT_EQ(dists.size(), 1);

        queue.MemcpyAsync(dRandomNumbers,
                          Span<const Vector2>(hRandomNumbers.cbegin(),
                                              hRandomNumbers.cend()));

        queue.IssueSaturatingKernel<KCSampleDist<Dist2D>>
        (
            "GTest SampleDist2D"sv,
            KernelIssueParams{.workCount = sampleCount},
            //
            dOutSamples,
            dRandomNumbers,
            Span<const Dist2D, 1>(dists.subspan(0, 1)),
            sampleCount
        );

        std::vector<SampleT<Vector2>> hOutSamples(sampleCount);
        queue.MemcpyAsync(Span<SampleT<Vector2>>(hOutSamples.begin(),
                                                 hOutSamples.end()),
                          ToConstSpan(dOutSamples));
        queue.Barrier().Wait();
        return std::move(hOutSamples);
    }
};

TEST(PiecewiseConstant2D, Uniform)
{
    static constexpr uint32_t SAMPLE_COUNT = 4096;
    // Testing with a 4k image (average case?)
    static constexpr Vector2ui SIZE = Vector2ui(3840, 2160);
    static constexpr size_t SIZE_LINEAR = SIZE.Multiply();
    std::vector<Float> hFunction(SIZE_LINEAR, Float{12.0});

    std::mt19937 rng(332);
    std::vector<Vector2> hRandomNumbers(SAMPLE_COUNT);
    std::uniform_real_distribution<Float> dist;
    std::for_each(hRandomNumbers.begin(), hRandomNumbers.end(),
                  [&](Vector2& rn)
    {
        rn[0] = dist(rng);
        rn[1] = dist(rng);
    });
    // Add boundary ranges
    hRandomNumbers[0] = Vector2::Zero();
    hRandomNumbers[1] = Vector2(std::nexttoward(Float(1), Float(0)));

    DistTester2D tester;
    auto hOutSamples = tester.GenSamplesIndex(hFunction, SIZE, hRandomNumbers);

    uint32_t i = 0;
    for(const auto& s : hOutSamples)
    {
        using namespace MathConstants;
        EXPECT_NEAR(s.pdf, Float{1}, VeryLargeEpsilon<Float>());

        // On uniform function, random numberss should match to the
        // sampled index
        Vector2 indexExpected = hRandomNumbers[i] * Vector2(SIZE);
        EXPECT_EQUAL_MRAY(s.sampledResult, indexExpected,
                          VeryLargeEpsilon<Float>());
        i++;
    }
}

TEST(PiecewiseConstant2D, ZeroVariance)
{
    static constexpr uint32_t SAMPLE_COUNT = 4096 * 4;
    // Testing with a 4k image (average case?)
    static constexpr Vector2ui SIZE = Vector2ui(3840, 2160);
    static constexpr size_t SIZE_LINEAR = SIZE.Multiply();

    // Function overall min/max
    static constexpr Float FUNCTION_MIN = 0;
    static constexpr Float FUNCTION_MAX = 10;
    // Simulating "sun" on an HDR image, adjacent
    //  couple of pixels will have these value
    static constexpr Float FUNCTION_PEAK = 2.0e4;
    // Manually tightened this by experimentation
    // It probably only works for the parameters above
    static constexpr Float GiganticEpsilon = Float(0.06);

    std::mt19937 rng(332);

    std::vector<Vector2> hRandomNumbers(SAMPLE_COUNT);
    std::uniform_real_distribution<Float> dist01;
    std::for_each(hRandomNumbers.begin(), hRandomNumbers.end(),
                  [&](Vector2& rn)
    {
        rn[0] = dist01(rng);
        rn[1] = dist01(rng);
    });
    // Add boundary ranges
    hRandomNumbers[0] = Vector2::Zero();
    hRandomNumbers[1] = Vector2(std::nexttoward(Float(1), Float(0)));


    std::vector<Float> hFunction(SIZE_LINEAR);
    std::uniform_real_distribution<Float> distF(FUNCTION_MIN, FUNCTION_MAX);
    std::for_each(hFunction.begin(), hFunction.end(),
                  [&](Float& f)
    {
        f = distF(rng);
    });
    // Add some peaks (pixel-wide sun maybe?)
    for(uint32_t i = 0; i < 2; i++)
    for(uint32_t j = 0; j < 2; j++)
    {
        Vector2ui middle = SIZE / 2 + Vector2ui(i, j);
        uint32_t midIndex = middle[1] * SIZE[0] + middle[0];
        assert(midIndex < SIZE_LINEAR);
        hFunction[midIndex] = FUNCTION_PEAK;
    }

    DistTester2D tester;
    auto hOutSamples = tester.GenSamplesIndex(hFunction, SIZE, hRandomNumbers);

    // Integrate the function
    Float total = std::reduce(hFunction.cbegin(), hFunction.cend(), Float{0});
    Vector2 dxy = Vector2(1) / Vector2(SIZE);
    Float integralExpected = total * dxy.Multiply();

    uint32_t i = 0;
    Float monteCarlo = 0;
    using namespace MathConstants;
    for(const auto& s : hOutSamples)
    {
        Vector2ui functionIndex = Vector2ui(s.sampledResult);
        uint32_t indexLinear = functionIndex[1] * SIZE[0] + functionIndex[0];
        ASSERT_LT(indexLinear, SIZE_LINEAR);
        Float f = hFunction[indexLinear];
        Float integralEstimate = f / s.pdf;

        EXPECT_NEAR(integralEstimate, integralExpected, GiganticEpsilon);
        monteCarlo += integralEstimate;
        i++;
    }
    // Check the Monte Carlo
    // Technically this should not be better since
    // variance comes from numerical precision.
    static constexpr Float SAMPLE_COUNT_RECIP = Float(1) / SAMPLE_COUNT;
    monteCarlo *= SAMPLE_COUNT_RECIP;
    EXPECT_NEAR(monteCarlo, integralExpected, GiganticEpsilon);
}

TEST(Linear, ZeroVariance)
{
    static constexpr uint32_t SAMPLE_COUNT = 128;
    static constexpr uint32_t FUNCTION_COUNT = 16;
    // Function overall min/max
    static constexpr Float FUNCTION_MIN = -10;
    static constexpr Float FUNCTION_MAX = 10;

    std::mt19937 rng(332);
    using UniformDist = std::uniform_real_distribution<Float>;

    UniformDist dist01;
    UniformDist distCD(FUNCTION_MIN, FUNCTION_MAX);
    for(uint32_t f = 0; f < FUNCTION_COUNT; f++)
    {
        Float c = distCD(rng);
        Float d = distCD(rng);
        const Float trapz = (c + d) * Float(0.5);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            Float xi = dist01(rng);

            auto result = Distributions::SampleLine(xi, c, d);
            // Evaluate the function
            Float eval = MathFunctions::Lerp(c, d, result.sampledResult);
            Float estimate = eval / result.pdf;
            // Since this is zero variance estimate,
            // the estimate should exactly match
            // actual integral
            EXPECT_FLOAT_EQ(trapz, estimate);
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        EXPECT_NEAR(trapz, total, MathConstants::LargeEpsilon<Float>());
    }
}