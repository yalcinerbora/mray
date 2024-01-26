#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Core/Vector.h"
#include "Core/Quaternion.h"
#include "Core/GraphicsFunctions.h"

#include "GTestWrappers.h"

TEST(GraphicsFunctionsTest, Reflect)
{
    using namespace GraphicsFunctions;

    {
        // Simple 2D Test
        Vector3 normal = Vector3::YAxis();
        Vector3 v = Vector3(0.5, 0.5, 0.0);
        Vector3 result = Reflect(normal, v);
        EXPECT_EQUAL_MRAY(result, Vector3(-0.5, 0.5, 0.0));
    }

    {
        // Orthogonal Test
        Vector3 normal = Vector3::YAxis();
        Vector3 v = Vector3::XAxis();
        Vector3 result = Reflect(normal, v);
        EXPECT_EQUAL_MRAY(result, -Vector3::XAxis());
    }
}

TEST(GraphicsFunctionsTest, Refract)
{
    using namespace GraphicsFunctions;
    {
        // Simple Test
        Vector3 normal = Vector3::YAxis();
        Vector3 v = Vector3(0.5, 0.5, 0.0).Normalize();
        Float eta0 = static_cast<Float>(1.0);
        Float eta1 = static_cast<Float>(1.3333);
        // From some online calculator
        Float Angle = (static_cast<Float>(32.0367) *
                       MathConstants::DegToRadCoef<Float>());
        // Assuming Core/Quaternion Tests are passed.
        Quaternion q(-Angle, Vector3::ZAxis());

        Vector3 result = Refract(normal, v, eta0, eta1).value();
        EXPECT_FLOAT_EQ(result.Length(), Float{1.0});
        EXPECT_NEAR(result.Dot(-normal), std::cos(Angle),
                    MathConstants::LargeEpsilon<Float>());
        EXPECT_EQUAL_MRAY(result, q.ApplyRotation(-normal),
                          MathConstants::VeryLargeEpsilon<Float>());
    }

    {
        // Total Internal Reflection
        Vector3 normal = Vector3::YAxis();
        Vector3 v = Vector3(0.5, 0.5, 0.0).Normalize();
        Float eta0 = static_cast<Float>(2.419);
        Float eta1 = static_cast<Float>(1.0);

        Optional<Vector3> result = Refract(normal, v, eta0, eta1);

        EXPECT_THROW(auto const a = result.value(), std::bad_optional_access);
    }
}

TEST(GraphicsFunctionsTest, Orient)
{
    using namespace GraphicsFunctions;
    {
        // Basic Test / No Change
        Vector3 normal = Vector3::YAxis();

        Vector3 v = Vector3(1.0, 2.0, 3.0);
        Vector3 result = Orient(v, normal);
        EXPECT_EQUAL_MRAY(result, v);
    }

    {
        // Basic Test / Change
        Vector3 normal = Vector3::YAxis();

        Vector3 v = Vector3(1.0, -2.0, 3.0);
        Vector3 result = Orient(v, normal);
        EXPECT_EQUAL_MRAY(result, -v);
    }

    {
        // Exactly Orthogonal
        Vector3 normal = Vector3::YAxis();
        Vector3 v = Vector3::ZAxis();
        Vector3 result = Orient(v, normal);
        EXPECT_EQUAL_MRAY(result, v);
    }
}

TEST(GraphicsFunctionsTest, SampleUniformHemisphere)
{
    using namespace GraphicsFunctions;
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
            total += (1 / sample.pdf);
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
            double functionVal = sample.sampledResult.Dot(Vector3::ZAxis());
            functionVal *= MathConstants::InvPi<double>();
            total += (functionVal / sample.pdf);
        }

        double result = total / double{Iterations};
        EXPECT_NEAR(result, 1.0, MathConstants::HugeEpsilon<double>());
    }
}

TEST(GraphicsFunctionsTest, SampleCosineHemisphere)
{
    using namespace GraphicsFunctions;
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
            total += (1 / sample.pdf);
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
            double functionVal = sample.sampledResult.Dot(Vector3::ZAxis());
            functionVal *= MathConstants::InvPi<double>();
            total += (functionVal / sample.pdf);
        }

        double result = total / double{Iterations};
        EXPECT_NEAR(result, 1.0, MathConstants::Epsilon<double>());
    }
}

TEST(GraphicsFunctionsTest, PDFUniformHemisphere)
{
    using namespace GraphicsFunctions;
    // As simple as it gets
    // Provided for completeness
    constexpr Float expected = MathConstants::InvPi<Float>() * Float{0.5};
    EXPECT_EQ(PDFUniformDirection(), expected);
}

TEST(GraphicsFunctionsTest, PDFCosineHemisphere)
{
    using namespace GraphicsFunctions;
    // As simple as it gets
    // Provided for completeness
    Vector3 v = Vector3(1, 2, 3).Normalize();
    Float expected = MathConstants::InvPi<Float>() * v.Dot(Vector3::ZAxis());
    EXPECT_EQ(PDFCosDirection(v), expected);
}

TEST(GraphicsFunctionsTest, GSOrthonormalize2D)
{
    {
        // Correction-like
        using namespace GraphicsFunctions;
        Vector3 x = Vector3::XAxis();
        Vector3 y = Vector3::YAxis();
        y += Vector3::XAxis() * MathConstants::LargeEpsilon<Float>();
        // Renormalize Y
        y = GSOrthonormalize(y, x);
        EXPECT_EQUAL_MRAY(y, Vector3::YAxis(), MathConstants::Epsilon<Float>());
    }

    {
        // Big difference
        using namespace GraphicsFunctions;
        Vector3 x = Vector3::XAxis();
        Vector3 y = Vector3(0.5, 0.5, 0.0).Normalize();
        // Renormalize Y
        y = GSOrthonormalize(y, x);
        EXPECT_EQUAL_MRAY(y, Vector3::YAxis(), MathConstants::Epsilon<Float>());
    }
}

TEST(GraphicsFunctionsTest, GSOrthonormalize3D)
{
    {
        // Correction-like
        using namespace GraphicsFunctions;
        Vector3 x = Vector3::XAxis();
        Vector3 y = Vector3::YAxis();
        Vector3 z = Vector3::ZAxis();
        x += Vector3::ZAxis() * MathConstants::LargeEpsilon<Float>();
        y += Vector3::XAxis() * MathConstants::LargeEpsilon<Float>();
        // Renormalize Y
        std::tie(x, y) = GSOrthonormalize(x, y, z);
        EXPECT_EQUAL_MRAY(y, Vector3::YAxis(), MathConstants::LargeEpsilon<Float>());
        EXPECT_EQUAL_MRAY(x, Vector3::XAxis(), MathConstants::LargeEpsilon<Float>());
    }

    {
        // Big difference
        using namespace GraphicsFunctions;
        Vector3 x = Vector3::XAxis();
        Vector3 y = Vector3(0.0, 0.5, 0.5).Normalize();
        Vector3 z = Vector3::ZAxis();
        // Renormalize Y
        std::tie(x, y) = GSOrthonormalize(x, y, z);
        EXPECT_EQUAL_MRAY(y, Vector3::YAxis(), MathConstants::Epsilon<Float>());
        EXPECT_EQUAL_MRAY(x, Vector3::XAxis(), MathConstants::Epsilon<Float>());
    }
}

TEST(GraphicsFunctionsTest, ComposeMorton2D_32Bit)
{
    using namespace GraphicsFunctions;

    {
        Vector2ui val = Vector2ui(0b1111111111, 0b0);
        uint32_t result = MortonCode::Compose2D<uint32_t>(val);
        EXPECT_EQ(result, 0b01010101010101010101);
    }

    {
        Vector2ui val = Vector2ui(0b0, 0b1111111111);
        uint32_t result = MortonCode::Compose2D<uint32_t>(val);
        EXPECT_EQ(result, 0b10101010101010101010);
    }
}

TEST(GraphicsFunctionsTest, DecomposeMorton2D_32Bit)
{
    using namespace GraphicsFunctions;

    {
        uint32_t val = 0b01010101010101010101;
        Vector2ui result = MortonCode::Decompose2D<uint32_t>(val);
        EXPECT_EQ(result, Vector2ui(0b1111111111, 0b0));
    }

    {

        uint32_t val = 0b10101010101010101010;
        Vector2ui result = MortonCode::Decompose2D<uint32_t>(val);
        EXPECT_EQ(result, Vector2ui(0b0, 0b1111111111));
    }
}

TEST(GraphicsFunctionsTest, ComposeMorton3D_32Bit)
{
    using namespace GraphicsFunctions;

    {
        Vector3ui val = Vector3ui(0b1111111111, 0b0, 0b0);
        uint32_t result = MortonCode::Compose3D<uint32_t>(val);
        EXPECT_EQ(result, 0b001001001001001001001001001001);
    }

    {
        Vector3ui val = Vector3ui(0b0, 0b1111111111, 0b0);
        uint32_t result = MortonCode::Compose3D<uint32_t>(val);
        EXPECT_EQ(result, 0b010010010010010010010010010010);
    }

    {
        Vector3ui val = Vector3ui(0b0, 0b0, 0b1111111111);
        uint32_t result = MortonCode::Compose3D<uint32_t>(val);
        EXPECT_EQ(result, 0b100100100100100100100100100100);
    }
}

TEST(GraphicsFunctionsTest, DecomposeMorton3D_32Bit)
{
    using namespace GraphicsFunctions;

    {
        uint32_t val = 0b001001001001001001001001001001;
        Vector3ui result = MortonCode::Decompose3D<uint32_t>(val);
        EXPECT_EQ(result, Vector3ui(0b1111111111, 0b0, 0b0));
    }

    {
        uint32_t val = 0b010010010010010010010010010010;
        Vector3ui result = MortonCode::Decompose3D<uint32_t>(val);
        EXPECT_EQ(result, Vector3ui(0b0, 0b1111111111, 0b0));
    }

    {
        uint32_t val = 0b100100100100100100100100100100;
        Vector3ui result = MortonCode::Decompose3D<uint32_t>(val);
        EXPECT_EQ(result, Vector3ui(0b0, 0b0, 0b1111111111));
    }
}

TEST(GraphicsFunctionsTest, ComposeMorton2D_64Bit)
{
    using namespace GraphicsFunctions;

    {
        Vector2ui val = Vector2ui(0b11111111111111111111, 0b0);
        uint64_t result = MortonCode::Compose2D<uint64_t>(val);
        EXPECT_EQ(result, 0b0101010101010101010101010101010101010101);
    }

    {
        Vector2ui val = Vector2ui(0b0, 0b11111111111111111111);
        uint64_t result = MortonCode::Compose2D<uint64_t>(val);
        EXPECT_EQ(result, 0b1010101010101010101010101010101010101010);
    }
}

TEST(GraphicsFunctionsTest, DecomposeMorton2D_64Bit)
{
    using namespace GraphicsFunctions;

    {
        uint64_t val = 0b0101010101010101010101010101010101010101;
        Vector2ui result = MortonCode::Decompose2D<uint64_t>(val);
        EXPECT_EQ(result, Vector2ui(0b11111111111111111111, 0b0));
    }

    {

        uint64_t val = 0b1010101010101010101010101010101010101010;
        Vector2ui result = MortonCode::Decompose2D<uint64_t>(val);
        EXPECT_EQ(result, Vector2ui(0b0, 0b11111111111111111111));
    }
}

TEST(GraphicsFunctionsTest, ComposeMorton3D_64Bit)
{
    using namespace GraphicsFunctions;

    {
        Vector3ui val = Vector3ui(0b11111111111111111111, 0b0, 0b0);
        uint64_t result = MortonCode::Compose3D<uint64_t>(val);
        EXPECT_EQ(result, 0x249249249249249);
    }

    {
        Vector3ui val = Vector3ui(0b0, 0b11111111111111111111, 0b0);
        uint64_t result = MortonCode::Compose3D<uint64_t>(val);
        EXPECT_EQ(result, 0x492492492492492);
    }

    {
        Vector3ui val = Vector3ui(0b0, 0b0, 0b11111111111111111111);
        uint64_t result = MortonCode::Compose3D<uint64_t>(val);
        EXPECT_EQ(result, 0x924924924924924);
    }
}

TEST(GraphicsFunctionsTest, DecomposeMorton3D_64Bit)
{
    using namespace GraphicsFunctions;

    {
        uint64_t val = 0x249249249249249;
        Vector3ui result = MortonCode::Decompose3D<uint64_t>(val);
        EXPECT_EQ(result, Vector3ui(0b11111111111111111111, 0b0, 0b0));
    }

    {
        uint64_t val = 0x492492492492492;
        Vector3ui result = MortonCode::Decompose3D<uint64_t>(val);
        EXPECT_EQ(result, Vector3ui(0b0, 0b11111111111111111111, 0b0));
    }

    {
        uint64_t val = 0x924924924924924;
        Vector3ui result = MortonCode::Decompose3D<uint64_t>(val);
        EXPECT_EQ(result, Vector3ui(0b0, 0b0, 0b11111111111111111111));
    }
}


TEST(GraphicsFunctionsTest, DISABLED_SphericalToCartesian)
{

}

TEST(GraphicsFunctionsTest, DISABLED_CartesianToSpherical)
{

}

TEST(GraphicsFunctionsTest, DISABLED_DirectionToCoOcto)
{

}

TEST(GraphicsFunctionsTest, DISABLED_CoOctoToDirection)
{

}

TEST(GraphicsFunctionsTest, DISABLED_CoOctoWrap)
{

}

TEST(GraphicsFunctionsTest, DISABLED_CoOctoWrapInt)
{

}