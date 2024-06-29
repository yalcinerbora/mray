#pragma once

#include "Core/Vector.h"
#include "Core/GraphicsFunctions.h"
#include "Core/MathFunctions.h"
#include "TracerTypes.h"

namespace BxDFFunctions
{

}

namespace MediumFunctions
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

namespace Distributions
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

namespace BxDFFunctions
{

}

namespace Distributions
{

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> BisectSample1(Float xi, Float)
{
    return {0, xi};
}

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> BisectSample2(Float xi, Vector2 weights,
                                    bool isAlreadyNorm)
{
    if(!isAlreadyNorm) weights[0] /= weights.Sum();
    //
    Float w = weights[0];
    return Pair<uint32_t, Float>
    {
        (xi < w) ? 0 : 1,
        //
        (xi < w) ? xi       / w
                 : (xi - w) / (Float(1) - w)
    };
}


template<uint32_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> BisectSample(Float xi, const Span<Float, N>& weights,
                                   bool isAlreadyNorm)
{
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
    return Pair<uint32_t, Float>
    {
        i, (xiScaled - weights[i]) / (weights[i + 1] - weights[i])
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> SampleGaussian(Float xi, Float sigma, Float mu)
{
    Float x = MathConstants::Sqrt2<Float>() * sigma;
    Float e = MathFunctions::InvErrFunc(Float(2) * xi - Float(1));
    x = x * e + mu;
    return SampleT<Float>
    {
        .sampledResult = x,
        .pdf = PDFGaussian(x, sigma, mu)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float PDFGaussian(Float x, Float sigma, Float mu)
{
    return MathFunctions::Gaussian(x, sigma, mu);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> SampleLine(Float xi, Float c, Float d)
{
    // https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Sampling_Using_the_Inversion_Method#SampleLinear
    Float normVal = Float(2) / (c + d);
    // Avoid divide by zero
    if(xi == 0)
    {
        return SampleT<Float>
        {
            .sampledResult = 0,
            .pdf = normVal * c
        };
    }
    Float denom = MathFunctions::Lerp(c * c, d * d, xi);
    denom = c + std::sqrt(denom);
    Float x = (c + d) * xi / denom;
    return SampleT<Float>
    {
        .sampledResult = x,
        .pdf = normVal * MathFunctions::Lerp(c, d, x)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float PDFLine(Float x, Float c, Float d)
{
    if(x < 0 && x > 1) return Float(0);
    Float normVal = Float(2) / (c + d);
    return normVal * MathFunctions::Lerp(c, d, x);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> SampleTent(Float xi, Float a, Float b)
{
    assert(a <= b);
    Float mid = (b - a) * Float(0.5);
    auto [index, localXi] = BisectSample2(xi, Vector2(0.5));
    SampleT<Float> result = (index == 0)
        ? SampleLine(localXi, 0, 1)
        : SampleLine(localXi, 1, 0);
    Float offset = (index == 0) ? a : mid;

    Float& x = result.sampledResult;
    x = x * mid + offset;
    result.pdf *= Float(0.5);
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float PDFTent(Float x, Float a, Float b)
{
    Float mid = (b - a) * Float(0.5);
    Float x01 = std::abs(x - mid);
    return PDFLine(x01, 1, 0) * Float(0.5);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> SampleUniformRange(Float xi, Float a, Float b)
{
    return SampleT<Float>
    {
        .sampledResult = xi * (b - a) + a,
        .pdf = Float(1) / (b - a)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float PDFUniformRange(Float x, Float a, Float b)
{
    if(x < a && x > b) return 0;
    return 1;
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
        .sampledResult = wI,
        .pdf = HenyeyGreensteinPhase(cosTheta, g)
    };
}


}