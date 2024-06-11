#pragma once

#include "Core/Vector.h"
#include "Core/GraphicsFunctions.h"
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

namespace BxDFFunctions
{

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