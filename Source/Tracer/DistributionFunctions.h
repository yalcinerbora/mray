#pragma once

#include "Core/Vector.h"
#include "Core/GraphicsFunctions.h"
#include "Core/MathFunctions.h"
#include "TracerTypes.h"

#include "Core/Log.h"

namespace Distributions::BxDF
{

}

namespace Distributions::Medium
{
    MRAY_HYBRID
    constexpr Spectrum  WavesToSpectrumCauchy(const SpectrumWaves& waves,
                                              const Vector2& coeffs);
    MRAY_HYBRID
    Float               HenyeyGreensteinPhase(Float cosTheta, Float g);

    MRAY_HYBRID
    SampleT<Vector3>    SampleHenyeyGreensteinPhase(const Vector3& wO, Float g,
                                                    const Vector2& xi);
}

namespace Distributions::Common
{
    MRAY_HYBRID
    Pair<uint32_t, Float>   BisectSample1(Float xi, Float weight);
    MRAY_HYBRID
    Pair<uint32_t, Float>   BisectSample2(Float xi, Vector2 weights,
                                          bool isAlreadyNorm = false);
    template<uint32_t N>
    MRAY_HYBRID
    Pair<uint32_t, Float>   BisectSample(Float xi, const Span<Float, N>& weights,
                                         bool isAlreadyNorm = false);

    MRAY_HYBRID
    SampleT<Float>  SampleGaussian(Float xi, Float sigma = Float(1),
                                   Float mu = Float(0));
    MRAY_HYBRID
    Float           PDFGaussian(Float x, Float sigma = Float(1),
                                Float mu = Float(0));

    MRAY_HYBRID
    SampleT<Float>  SampleLine(Float xi, Float c, Float d);
    MRAY_HYBRID
    Float           PDFLine(Float x, Float c, Float d);

    MRAY_HYBRID
    SampleT<Float>  SampleTent(Float xi, Float a, Float b);
    MRAY_HYBRID
    Float           PDFTent(Float x, Float a, Float b);

    MRAY_HYBRID
    SampleT<Float>  SampleUniformRange(Float xi, Float a, Float b);
    MRAY_HYBRID
    Float           PDFUniformRange(Float x, Float a, Float b);
    //
    MRAY_HYBRID
    SampleT<Vector3>    SampleCosDirection(const Vector2& xi);
    MRAY_HYBRID
    constexpr Float     PDFCosDirection(const Vector3& v,
                                        const Vector3& n = Vector3::ZAxis());
    MRAY_HYBRID
    SampleT<Vector3>    SampleUniformDirection(const Vector2& xi);
    MRAY_HYBRID
    constexpr Float     PDFUniformDirection();
}

namespace Distributions::MIS
{
    template<uint32_t N>
    MRAY_HYBRID
    Float BalanceCancelled(const Span<Float, N>& pdfs,
                           const Span<Float, N>& weights);

    template<uint32_t N>
    MRAY_HYBRID
    Float Balance(uint32_t pdfIndex,
                  const Span<Float, N>& pdfs,
                  const Span<Float, N>& weights);
}

namespace Distributions
{

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> Common::BisectSample1(Float xi, Float)
{
    return {0, xi};
}

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> Common::BisectSample2(Float xi, Vector2 weights,
                                            bool isAlreadyNorm)
{
    using MathFunctions::PrevFloat;
    if(!isAlreadyNorm) weights[0] /= weights.Sum();
    //
    Float w = weights[0];
    uint32_t i = (xi < w) ? 0 : 1;
    Float localXi = (xi < w)
            ? xi / w
            : (xi - w) / (Float(1) - w);
    localXi = std::min(localXi, PrevFloat<Float>(1));
    assert(localXi >= 0 && localXi < 1);
    return Pair(i, localXi);
}

template<uint32_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> Common::BisectSample(Float xi, const Span<Float, N>& weights,
                                           bool isAlreadyNorm)
{
    using MathFunctions::PrevFloat;
    auto Reduce = [weights]() -> Float
    {
        Float r = 0;
        UNROLL_LOOP
        for(uint32_t i = 1; i < N; i++)
            r += weights[i];
        return r;
    };
    auto Find = [weights](Float xiScaled) -> uint32_t
    {
        Float sum = 0;
        UNROLL_LOOP
        for(uint32_t i = 0; i < N; i++)
        {
            sum += weights[i];
            if(xiScaled < sum) return i;
        }
        return N;
    };
    // Since N is compile time constant, assuming N is small.
    // If not it is better to use PWC 1D distribution.
    // So we don't do binary search here and yolo linear search.
    Float sum = (isAlreadyNorm) ? Float(1) : Reduce();
    Float xiScaled = xi * Float(1) / sum;
    uint32_t i = Find(xiScaled);
    assert(i != N);
    Float localXi = (xiScaled - weights[i]) / (weights[i + 1] - weights[i]);
    localXi = std::min(localXi, PrevFloat<Float>(1));
    assert(localXi >= 0 && localXi < 1);
    return Pair(i, localXi);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleGaussian(Float xi, Float sigma, Float mu)
{
    Float x = MathConstants::Sqrt2<Float>() * sigma;
    Float e = MathFunctions::InvErrFunc(Float(2) * xi - Float(1));
    x = x * e + mu;

    // Erf can be -+inf, when xi is near zero or one
    // Just clamp to %99.95 estimate of the actual integral
    if(MathFunctions::IsInf(e))
    {
        Float minMax = Float(3.5) * sigma;
        x = MathFunctions::Clamp(x, -minMax, minMax);
    }

    return SampleT<Float>
    {
        .value = x,
        .pdf = PDFGaussian(x, sigma, mu)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFGaussian(Float x, Float sigma, Float mu)
{
    return MathFunctions::Gaussian(x, sigma, mu);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleLine(Float xi, Float c, Float d)
{
    using namespace MathFunctions;
    // https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Sampling_Using_the_Inversion_Method#SampleLinear
    Float normVal = Float(2) / (c + d);
    // Avoid divide by zero
    const Float epsilon = NextFloat<Float>(0);
    if(c == 0 && xi == 0)
        return SampleT{epsilon, normVal * epsilon};

    Float denom = Lerp(c * c, d * d, xi);
    denom = c + std::sqrt(denom);
    Float x = (c + d) * xi / denom;
    return SampleT<Float>
    {
        .value = x,
        .pdf = normVal * Lerp(c, d, x)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFLine(Float x, Float c, Float d)
{
    if(x < 0 && x > 1) return Float(0);
    Float normVal = Float(2) / (c + d);
    return normVal * MathFunctions::Lerp(c, d, x);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleTent(Float xi, Float a, Float b)
{
    // Dirac delta like, return as if dirac delta
    if(b - a < MathConstants::Epsilon<Float>())
        return {Float(0), Float(1) / (b - a)};

    using MathFunctions::PrevFloat;
    assert(a < 0 && b > 0);
    auto [index, localXi] = BisectSample2(xi, Vector2(-a, b), false);
    localXi = (index == 0) ? (PrevFloat<Float>(1) - localXi) : localXi;
    SampleT<Float> result = SampleLine(localXi, 1, 0);
    Float& x = result.value;
    x = (index == 0) ? (x * a) : (x * b);
    result.pdf /= (b - a);
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFTent(Float x, Float a, Float b)
{
    Float mid = (b - a) * Float(0.5);
    Float x01 = std::abs(x - mid);
    return PDFLine(x01, 1, 0) * Float(0.5);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleUniformRange(Float xi, Float a, Float b)
{
    return SampleT<Float>
    {
        .value = xi * (b - a) + a,
        .pdf = Float(1) / (b - a)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFUniformRange(Float x, Float a, Float b)
{
    if(x < a && x > b) return 0;
    return 1;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> Common::SampleCosDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using MathFunctions::SqrtMax;

    // Generated direction is on unit space (+Z oriented hemisphere)
    Float xi1Angle = Float{2} * Pi<Float>() * xi[1];
    Float xi0Sqrt = std::sqrt(xi[0]);

    Vector3 dir;
    dir[0] = xi0Sqrt * std::cos(xi1Angle);
    dir[1] = xi0Sqrt * std::sin(xi1Angle);
    dir[2] = SqrtMax(Float{1} - Vector2(dir).Dot(Vector2(dir)));

    // Fast tangent space dot product and domain constant
    Float pdf = dir[2] * InvPi<Float>();

    // Finally the result!
    return SampleT<Vector3>
    {
        .value = dir,
        .pdf = pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Float Common::PDFCosDirection(const Vector3& v, const Vector3& n)
{
    Float pdf = n.Dot(v) * MathConstants::InvPi<Float>();
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> Common::SampleUniformDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using MathFunctions::SqrtMax;

    Float xi0Sqrt = SqrtMax(Float{1} - xi[0] * xi[0]);
    Float xi1Angle = 2 * Pi<Float>() * xi[1];

    Vector3 dir;
    dir[0] = xi0Sqrt * std::cos(xi1Angle);
    dir[1] = xi0Sqrt * std::sin(xi1Angle);
    dir[2] = xi[0];

    // Uniform pdf is invariant
    constexpr Float pdf = InvPi<Float>() * Float{0.5};
    return SampleT<Vector3>
    {
        .value = dir,
        .pdf = pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Float Common::PDFUniformDirection()
{
    return MathConstants::InvPi<Float>() * Float{0.5};
}

template<uint32_t N>
MRAY_HYBRID
Float MIS::BalanceCancelled(const Span<Float, N>& pdfs,
                            const Span<Float, N>& weights)
{
    Float result = Float(0);
    UNROLL_LOOP
    for(uint32_t i = 0; i < N; i++)
        result += pdfs[i] * weights[i];
    return result;
}

template<uint32_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MIS::Balance(uint32_t pdfIndex,
                   const Span<Float, N>& pdfs,
                   const Span<Float, N>& weights)
{
    Float denom = BalanceCancelled(pdfIndex, pdfs, weights);
    return weights[pdfIndex] * pdfs[pdfIndex] / denom;
}

}

namespace MediumFunctions
{

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Spectrum WavesToSpectrumCauchy(const SpectrumWaves& waves,
                                         const Vector2& coeffs)
{
    Spectrum result;
    UNROLL_LOOP
    for(uint32_t i = 0; i < SpectrumWaves::Dims; i++)
    {
        result[i] = coeffs[0] + coeffs[1] / (waves[0] * waves[0]);
    }
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float HenyeyGreensteinPhase(Float cosTheta, Float g)
{
    // From the PBR book
    // https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions#HenyeyGreenstein

    Float gSqr = g * g;
    Float nom = (Float(1) - gSqr) * MathConstants::Inv4Pi<Float>();
    // https://www.desmos.com/calculator/7ogsbedc2r
    // the sqrtIn is never technically zero
    // due to numerical errors it can be zero
    Float sqrtIn = Float(1) + gSqr + cosTheta;
    Float denom = MathFunctions::SqrtMax(sqrtIn);
    denom *= sqrtIn;
    return nom / denom;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> SampleHenyeyGreensteinPhase(const Vector3& wO, Float g,
                                             const Vector2& xi)
{
    using namespace MathFunctions;
    // From the PBR book
    // https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions#HenyeyGreenstein
    // Phi is easy
    constexpr Float TwoPi = MathConstants::Pi<Float>() * Float(2);
    Float phi = TwoPi * xi[0];
    Vector2 sinCosPhi = Vector2(SinCos(phi));

    // Theta is involved
    Float gSqr = g * g;
    Float nom = Float(1) - gSqr;
    Float denom = Float(1) + g - Float(2) * g * xi[1];
    Float hg = Float(1) + gSqr - nom / denom;
    hg *= Float(-1) / (Float(2) * g);
    // https://www.desmos.com/calculator/7ogsbedc2r
    // cosTheta when g is near zero should be inverted maybe?
    bool isNearZero = (std::abs(g) < MathConstants::VeryLargeEpsilon<Float>());
    Float cosTheta = (isNearZero)
                        ? (Float(2) * xi[1] - Float(1))
                        : hg;
    Float sinTheta = SqrtMax(Float(1) - cosTheta * cosTheta);
    Vector2 sinCosTheta = Vector2(sinTheta, cosTheta);

    // This is in unit space, convert to wO's space
    // TODO: Check this
    Vector3 wI = GraphicsFunctions::UnitSphericalToCartesian(sinCosPhi, sinCosTheta);
    Quaternion rot = Quaternion::RotationBetweenZAxis(wO);
    wI = rot.ApplyRotation(wI);

    return SampleT<Vector3>
    {
        .value = wI,
        .pdf = HenyeyGreensteinPhase(cosTheta, g)
    };
}


}