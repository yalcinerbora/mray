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
template<class Dist2D, bool DoUV>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSampleDist(Span<SampleT<Vector2>> dOutSamples,
                  Span<Float> dOutPdfs,
                  Span<const Vector2> dRandomNumbers,
                  Span<const Dist2D, 1> dDist,
                  uint32_t sampleCount)
{
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        SampleT<Vector2> sample;
        Float pdf;
        if constexpr(DoUV)
        {
            sample = dDist[0].SampleUV(dRandomNumbers[i]);
            pdf = dDist[0].PdfUV(sample.value);
        }
        else
        {
            sample = dDist[0].SampleIndex(dRandomNumbers[i]);
            pdf = dDist[0].PdfIndex(sample.value);
        }
        dOutSamples[i] = sample;
        dOutPdfs[i] = pdf;
    }
}

struct DistTester2D
{
    using Dist2D = typename DistributionGroupPwC2D::Distribution2D;
    using DistData = typename DistributionGroupPwC2D::DistDataConst;

    GPUSystem system;
    DistributionGroupPwC2D distGroup;

    DistTester2D() : distGroup(system) {}

    template<bool DoUV>
    Pair<std::vector<SampleT<Vector2>>,
         std::vector<Float>>
    GenSamples(const std::vector<Float>& hFunction,
               const Vector2ui& size,
               const std::vector<Vector2>& hRandomNumbers)
    {
        using namespace std::literals;
        const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);

        uint32_t sampleCount = static_cast<uint32_t>(hRandomNumbers.size());
        uint32_t sizeLinear = size.Multiply();

        Span<Float> dOutPdfs;
        Span<SampleT<Vector2>> dOutSamples;
        Span<Vector2> dRandomNumbers;
        Span<Float> dFunction;
        DeviceMemory mem({&system.BestDevice()}, 32_MiB, 128_MiB);
        MemAlloc::AllocateMultiData(std::tie(dFunction, dRandomNumbers,
                                             dOutSamples, dOutPdfs),
                                    mem,
                                    {sizeLinear, sampleCount, sampleCount, sampleCount});
        queue.MemcpyAsync(dFunction, Span<const Float>(hFunction.cbegin(), hFunction.cend()));

        uint32_t id = distGroup.Reserve(size);
        distGroup.Commit();
        distGroup.Construct(id, dFunction, queue);

        Span<const Dist2D> dists = distGroup.DeviceDistributions();
        EXPECT_EQ(dists.size(), 1);

        queue.MemcpyAsync(dRandomNumbers,
                          Span<const Vector2>(hRandomNumbers.cbegin(),
                                              hRandomNumbers.cend()));

        queue.IssueSaturatingKernel<KCSampleDist<Dist2D, DoUV>>
        (
            "GTest SampleDist2D"sv,
            KernelIssueParams{.workCount = sampleCount},
            //
            dOutSamples,
            dOutPdfs,
            dRandomNumbers,
            Span<const Dist2D, 1>(dists.subspan(0, 1)),
            sampleCount
        );

        using Result = Pair<std::vector<SampleT<Vector2>>, std::vector<Float>>;
        Result result;
        result.first.resize(sampleCount);
        result.second.resize(sampleCount);

        queue.MemcpyAsync(Span(result.first), ToConstSpan(dOutSamples));
        queue.MemcpyAsync(Span(result.second), ToConstSpan(dOutPdfs));
        queue.Barrier().Wait();
        return result;
    }
};

TEST(Dist_PiecewiseConstant2D, Uniform)
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

    {
        DistTester2D tester;
        auto [hOutSamples, hOutPdfs]
            = tester.GenSamples<false>(hFunction, SIZE, hRandomNumbers);

        for(size_t i = 0; i < hOutSamples.size(); i++)
        {
            const auto& s = hOutSamples[i];
            const auto& pdf = hOutPdfs[i];

            using namespace MathConstants;
            EXPECT_NEAR(s.pdf, Float{1}, VeryLargeEpsilon<Float>());
            EXPECT_FLOAT_EQ(s.pdf, pdf);

            // On uniform function, random numberss should match to the
            // sampled index
            Vector2 indexExpected = hRandomNumbers[i] * Vector2(SIZE);
            EXPECT_EQUAL_MRAY(s.value, indexExpected,
                              VeryLargeEpsilon<Float>());
        }
    }

    {
        DistTester2D tester;
        auto [hOutSamples, hOutPdfs]
            = tester.GenSamples<true>(hFunction, SIZE, hRandomNumbers);

        for(size_t i = 0; i < hOutSamples.size(); i++)
        {
            const auto& s = hOutSamples[i];
            const auto& pdf = hOutPdfs[i];

            using namespace MathConstants;
            EXPECT_NEAR(s.pdf, Float{1}, VeryLargeEpsilon<Float>());
            EXPECT_FLOAT_EQ(s.pdf, pdf);

            if(s.pdf != pdf)
                __debugbreak();


            // On uniform function, random numberss should match to the
            // sampled index
            Vector2 expandedValue = s.value * Vector2(SIZE);
            Vector2 indexExpected = hRandomNumbers[i] * Vector2(SIZE);
            EXPECT_EQUAL_MRAY(expandedValue, indexExpected,
                              VeryLargeEpsilon<Float>());
        }
    }
}

TEST(Dist_PiecewiseConstant2D, ZeroVariance)
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
    auto [hOutSamples, hOutPdfs]
        = tester.GenSamples<false>(hFunction, SIZE, hRandomNumbers);

    // Integrate the function
    Float total = std::reduce(hFunction.cbegin(), hFunction.cend(), Float{0});
    Vector2 dxy = Vector2(1) / Vector2(SIZE);
    Float integralExpected = total * dxy.Multiply();

    Float monteCarlo = 0;
    using namespace MathConstants;
    for(size_t i = 0; i < hOutSamples.size(); i++)
    {
        const auto& s = hOutSamples[i];
        const auto& pdf = hOutPdfs[i];

        Vector2ui functionIndex = Vector2ui(s.value);
        uint32_t indexLinear = functionIndex[1] * SIZE[0] + functionIndex[0];
        ASSERT_LT(indexLinear, SIZE_LINEAR);
        Float f = hFunction[indexLinear];
        Float integralEstimate = f / s.pdf;

        EXPECT_FLOAT_EQ(s.pdf, pdf);
        EXPECT_NEAR(integralEstimate, integralExpected, GiganticEpsilon);
        monteCarlo += integralEstimate;
    }
    // Check the Monte Carlo
    // Technically this should not be better since
    // variance comes from numerical precision.
    static constexpr Float SAMPLE_COUNT_RECIP = Float(1) / SAMPLE_COUNT;
    monteCarlo *= SAMPLE_COUNT_RECIP;
    EXPECT_NEAR(monteCarlo, integralExpected, GiganticEpsilon);
}

TEST(Dist_Linear, ZeroVariance)
{
    using namespace MathConstants;
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
        Float c = 0;
        Float d = 1;
        if(f == 1)
            std::swap(c, d);
        else if(c > 1)
        {
            c = distCD(rng);
            d = distCD(rng);
        }
        const Float trapz = (c + d) * Float(0.5);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Float xi;
            if(i == 0) xi = Float(0);
            else if(i == 1) xi = Math::PrevFloat<Float>(1);
            else xi = dist01(rng);

            using namespace Distribution;
            auto result = Common::SampleLine(xi, c, d);
            // Check pdf from the function
            Float pdfFromFunc = Common::PDFLine(result.value, c, d);
            EXPECT_NEAR(pdfFromFunc, result.pdf, VeryLargeEpsilon<Float>());
            EXPECT_GE(result.value, 0);
            EXPECT_LT(result.value, 1);
            // Evaluate the function
            Float eval = Math::Lerp(c, d, result.value);
            Float estimate = eval / result.pdf;
            // Since this is zero variance estimate,
            // the estimate should exactly match
            // actual integral
            EXPECT_FLOAT_EQ(trapz, estimate);
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        EXPECT_NEAR(trapz, total, LargeEpsilon<Float>());
    }
}

TEST(Dist_Gaussian, ZeroVariance)
{
    using namespace MathConstants;
    static constexpr uint32_t SAMPLE_COUNT = 128;
    static constexpr uint32_t FUNCTION_COUNT = 16;
    // Function overall min/max
    static constexpr Float FUNCTION_MIN = -10;
    static constexpr Float FUNCTION_MAX = 10;

    std::mt19937 rng(332);
    using UniformDist = std::uniform_real_distribution<Float>;

    UniformDist dist01;
    UniformDist distMean(FUNCTION_MIN, FUNCTION_MAX);
    UniformDist distSigma(0, FUNCTION_MAX);
    for(uint32_t f = 0; f < FUNCTION_COUNT; f++)
    {
        Float mean = (f == 0) ? Float(0) : distMean(rng);
        Float sigma = (f == 0)
                        ? Epsilon<Float>()
                        : distSigma(rng);

        const Float integral = Float(1);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Float xi;
            if(i == 0) xi = Float(0);
            else if(i == 1) xi = Math::PrevFloat<Float>(1);
            else xi = dist01(rng);

            using namespace Distribution;
            auto result = Common::SampleGaussian(xi, sigma, mean);
            // Check pdf from the function
            Float pdfFromFunc = Common::PDFGaussian(result.value, sigma, mean);
            EXPECT_NEAR(pdfFromFunc, result.pdf, VeryLargeEpsilon<Float>());
            // Evaluate the function
            Float eval = Math::Gaussian(result.value,
                                                 sigma, mean);
            Float estimate = eval / result.pdf;
            // Since this is zero variance estimate,
            // the estimate should exactly match
            // actual integral.
            // For gaussian it will require EXPECT_NEAR
            // though since it is numerically comples
            EXPECT_FLOAT_EQ(integral, estimate);
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        EXPECT_NEAR(integral, total, LargeEpsilon<Float>());
    }
}

TEST(Dist_Gaussian2D, ZeroVariance)
{
    using namespace MathConstants;
    static constexpr uint32_t SAMPLE_COUNT = 128;
    static constexpr uint32_t FUNCTION_COUNT = 16;
    // Function overall min/max
    static constexpr Float FUNCTION_MIN = -10;
    static constexpr Float FUNCTION_MAX = 10;

    std::mt19937 rng(332);
    using UniformDist = std::uniform_real_distribution<Float>;

    UniformDist dist01;
    UniformDist distMean(FUNCTION_MIN, FUNCTION_MAX);
    UniformDist distSigma(0, FUNCTION_MAX);
    for(uint32_t f = 0; f < FUNCTION_COUNT; f++)
    {
        Vector2 mean = (f == 0)
                        ? Vector2::Zero()
                        : Vector2(distMean(rng),
                                  distMean(rng));
        Float sigma = (f == 0)
                        ? Epsilon<Float>()
                        : distSigma(rng);

        const Float integral = Float(1);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Vector2 xi;
            if(i == 0) xi = Vector2::Zero();
            else if(i == 1) xi = Vector2(Math::PrevFloat<Float>(1));
            else xi = Vector2(dist01(rng), dist01(rng));

            using namespace Distribution;
            auto result = Common::SampleGaussian2D(xi, sigma, mean);
            // Check pdf from the function
            Float pdfFromFunc = Common::PDFGaussian2D(result.value, sigma, mean);
            EXPECT_NEAR(pdfFromFunc, result.pdf, VeryLargeEpsilon<Float>());
            // Evaluate the function
            using Math::Gaussian;
            Float eval = (Gaussian(result.value[0], sigma, mean[0]) *
                          Gaussian(result.value[1], sigma, mean[1]));
            Float estimate = eval / result.pdf;
            // Since this is zero variance estimate,
            // the estimate should exactly match
            // actual integral.
            // For gaussian it will require EXPECT_NEAR
            // though since it is numerically comples
            EXPECT_FLOAT_EQ(integral, estimate);
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        EXPECT_NEAR(integral, total, LargeEpsilon<Float>());
    }
}

TEST(Dist_Tent, ZeroVariance)
{
    using namespace MathConstants;
    static constexpr uint32_t SAMPLE_COUNT = 128;
    static constexpr uint32_t FUNCTION_COUNT = 16;
    // Function overall min/max
    static constexpr Float FUNCTION_MIN = -5;
    static constexpr Float FUNCTION_MAX = 5;

    std::mt19937 rng(332);
    using UniformDist = std::uniform_real_distribution<Float>;

    UniformDist dist01;
    UniformDist distA(FUNCTION_MIN, 0);
    UniformDist distB(0, FUNCTION_MAX);
    for(uint32_t f = 0; f < FUNCTION_COUNT; f++)
    {
        Float a, b;
        if(f == 0)
        {
            a = -Epsilon<Float>();
            b = Epsilon<Float>();
        }
        else
        {
            a = distA(rng);
            b = distB(rng);
        }
        const Float integral = (b - a) * Float(0.5);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Float xi;
            if(i == 0) xi = Float(0);
            else if(i == 1) xi = Math::PrevFloat<Float>(1);
            else xi = dist01(rng);

            using namespace Distribution;
            auto result = Common::SampleTent(xi, a, b);
            // Check pdf from the function
            Float pdfFromFunc = Common::PDFTent(result.value, a, b);
            EXPECT_NEAR(pdfFromFunc, result.pdf, VeryLargeEpsilon<Float>());

            EXPECT_GT(result.value, a);
            EXPECT_LT(result.value, b);
            // Evaluate the function
            Float x = result.value;
            Float t = (x < 0) ? (x / a) : (x / b);

            Float eval = Math::Lerp<Float>(1, 0, t);
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

TEST(Dist_Uniform, ZeroVariance)
{
    static constexpr uint32_t SAMPLE_COUNT = 128;
    static constexpr uint32_t FUNCTION_COUNT = 16;
    // Function overall min/max
    static constexpr Float FUNCTION_MIN = -5;
    static constexpr Float FUNCTION_MAX = 5;

    std::mt19937 rng(332);
    using UniformDist = std::uniform_real_distribution<Float>;

    UniformDist dist01;
    UniformDist distA(FUNCTION_MIN, 0);
    UniformDist distB(0, FUNCTION_MAX);
    for(uint32_t f = 0; f < FUNCTION_COUNT; f++)
    {
        Float a, b;
        if(f == 0)
        {
            a = -MathConstants::Epsilon<Float>();
            b = MathConstants::Epsilon<Float>();
        }
        else
        {
            a = distA(rng);
            b = distB(rng);
        }
        const Float integral = (b - a);

        Float estimateTotal = 0;
        for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
        {
            // Put some edge cases to first two samples
            static_assert(SAMPLE_COUNT >= 2,
                          "At least two samples should be checked!");
            Float xi;
            if(i == 0) xi = Float(0);
            else if(i == 1) xi = Math::PrevFloat<Float>(1);
            else xi = dist01(rng);

            using namespace Distribution;
            auto result = Common::SampleUniformRange(xi, a, b);
            EXPECT_GE(result.value, a);
            EXPECT_LT(result.value, b);
            // Evaluate the function
            Float x = result.value;
            Float t = (x < 0) ? (x / -a) : (x / b);
            t = (x < 0) ? (x / a) : (x / b);
            t = std::abs(t);

            Float eval = Float(1);
            Float estimate = eval / result.pdf;
            // Since this is zero variance estimate,
            // the estimate should exactly match
            // actual integral.
            EXPECT_FLOAT_EQ(integral, estimate);
            estimateTotal += estimate;
        }
        Float total = estimateTotal / Float(SAMPLE_COUNT);
        EXPECT_NEAR(integral, total, MathConstants::LargeEpsilon<Float>());
    }
}

TEST(Dist_UniformHemisphere, Sample)
{
    using Distribution::Common::SampleUniformDirection;
    static constexpr uint32_t Iterations = 50'000;

    {
        std::mt19937 rng0(123), rng1(321);
        std::uniform_real_distribution<Float> dist0;
        std::uniform_real_distribution<Float> dist1;

        // Estimate Surface area
        double total = double{0};
        for(uint32_t i = 0; i < Iterations; i++)
        {
            Vector2 xi(dist0(rng0), dist1(rng1));
            SampleT<Vector3> sample = SampleUniformDirection(xi);
            // Integral of sin(theta) d(omega)
            total += static_cast<double>((1 / sample.pdf));
        }

        double result = total / double{Iterations};
        constexpr double expected = MathConstants::Pi<double>() * 2.0;
        EXPECT_NEAR(result, expected, MathConstants::LargeEpsilon<double>());
    }

    {
        std::mt19937 rng0(123), rng1(321);
        std::uniform_real_distribution<Float> dist0;
        std::uniform_real_distribution<Float> dist1;

        // Furnace test
        double total = double{0};
        for(uint32_t i = 0; i < Iterations; i++)
        {
            Vector2 xi(dist0(rng0), dist1(rng1));
            SampleT<Vector3> sample = SampleUniformDirection(xi);
            // Integral of cos(theta) d(omega)
            double functionVal = static_cast<double>(sample.value.Dot(Vector3::ZAxis()));
            functionVal *= MathConstants::InvPi<double>();
            total += (functionVal / static_cast<double>(sample.pdf));
        }

        double result = total / double{Iterations};
        EXPECT_NEAR(result, 1.0, MathConstants::HugeEpsilon<double>());
    }
}

TEST(Dist_UniformHemisphere, PDF)
{
    using Distribution::Common::PDFUniformDirection;
    // As simple as it gets
    // Provided for completeness
    constexpr Float expected = MathConstants::InvPi<Float>() * Float{ 0.5 };
    EXPECT_EQ(PDFUniformDirection(), expected);
}

TEST(Dist_CosineHemisphere, Sample)
{
    using namespace Distribution::Common;
    static constexpr uint32_t Iterations = 50'000;
    {
        std::mt19937 rng0(123), rng1(321);
        std::uniform_real_distribution<Float> dist0;
        std::uniform_real_distribution<Float> dist1;

        // Estimate Surface area
        double total = double{0};
        for(uint32_t i = 0; i < Iterations; i++)
        {
            Vector2 xi(dist0(rng0), dist1(rng1));
            SampleT<Vector3> sample = SampleUniformDirection(xi);
            // Integral of sin(theta) d(omega)
            total += static_cast<double>(1 / sample.pdf);
        }

        double result = total / double{Iterations};
        constexpr double expected = MathConstants::Pi<double>() * 2.0;
        EXPECT_NEAR(result, expected, MathConstants::LargeEpsilon<double>());
    }

    {
        std::mt19937 rng0(123), rng1(321);
        std::uniform_real_distribution<Float> dist0;
        std::uniform_real_distribution<Float> dist1;

        // Furnace test
        double total = double{0};
        for(uint32_t i = 0; i < Iterations; i++)
        {
            Vector2 xi(dist0(rng0), dist1(rng1));
            SampleT<Vector3> sample = SampleCosDirection(xi);
            // Integral of cos(theta) d(omega)
            double functionVal = static_cast<double>(sample.value.Dot(Vector3::ZAxis()));
            functionVal *= MathConstants::InvPi<double>();
            total += (functionVal / static_cast<double>(sample.pdf));
        }

        double result = total / double{Iterations};
        EXPECT_NEAR(result, 1.0, MathConstants::Epsilon<double>());
    }
}

TEST(Dist_CosineHemisphere, PDF)
{
    using Distribution::Common::PDFCosDirection;
    // As simple as it gets
    // Provided for completeness
    Vector3 v = Vector3(1, 2, 3).Normalize();
    Float expected = MathConstants::InvPi<Float>() * v.Dot(Vector3::ZAxis());
    EXPECT_EQ(PDFCosDirection(v), expected);
}
