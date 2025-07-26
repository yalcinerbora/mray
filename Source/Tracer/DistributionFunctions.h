#pragma once

#include "Core/Vector.h"
#include "Core/GraphicsFunctions.h"
#include "Core/Math.h"
#include "TracerTypes.h"

namespace Distribution::BxDF
{
    MRAY_HYBRID
    Float FresnelDielectric(Float cosFront, Float etaFront, Float etaBack);

    template<VectorC T>
    MRAY_HYBRID
    Float FresnelConductor(Float cosFront, const T& etaFront, const T& etaBack);

    MRAY_HYBRID
    Float DGGX(Float NdH, Float alpha);

    MRAY_HYBRID
    Float LambdaSmith(const Vector3& vec, Float alpha);

    MRAY_HYBRID
    Float GSmithSingle(const Vector3& vec, Float alpha);

    MRAY_HYBRID
    Float GSmithCorrelated(const Vector3& wO, const Vector3& wI,
                           Float alpha);

    MRAY_HYBRID
    Float GSmithSeparable(const Vector3& wO, const Vector3& wI,
                          Float alpha);

    MRAY_HYBRID
    Float GSchlick(Float cosTheta, Float alpha);

    MRAY_HYBRID
    Float GeomGGX(Float cosTheta, Float alpha);

    MRAY_HYBRID
    Spectrum FSchlick(Float VdH, const Spectrum& f0);

    MRAY_HYBRID
    Float VNDFGGXSmithPDF(const Vector3& V, const Vector3& H, Float alpha);

    MRAY_HYBRID
    SampleT<Vector3> VNDFGGXSmithSample(const Vector3& V, Float alpha,
                                        const Vector2& xi);

    MRAY_HYBRID
    Float BurleyDiffuseCorrection(Float NdL, Float NdV, Float LdH, Float roughness);
}

namespace Distribution::Medium
{
    MRAY_HYBRID
    constexpr Spectrum  WavesToSpectrumCauchy(const SpectrumWaves& waves,
                                              const Vector3& coeffs);
    MRAY_HYBRID
    Float               HenyeyGreensteinPhase(Float cosTheta, Float g);

    MRAY_HYBRID
    SampleT<Vector3>    SampleHenyeyGreensteinPhase(const Vector3& wO, Float g,
                                                    const Vector2& xi);
}

namespace Distribution::Common
{
    template<VectorOrFloatC  T>
    MRAY_HYBRID T       DivideByPDF(T, Float pdf);
    MRAY_HYBRID Float   DotN(Vector3);

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

    // TODO: Only providing isotropic version
    // here, anisotropic version may not be useful
    // We may provide a spherical version later.
    MRAY_HYBRID
    SampleT<Vector2>    SampleGaussian2D(Vector2 xi, Float sigma = Float(1),
                                         Vector2 mu = Vector2::Zero());
    MRAY_HYBRID
    Float               PDFGaussian2D(Vector2 xy, Float sigma = Float(1),
                                      Vector2 mu = Vector2::Zero());

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
    constexpr Float     PDFCosDirection(const Vector3& v, const Vector3& n);
    MRAY_HYBRID
    constexpr Float     PDFCosDirection(const Vector3& v);
    MRAY_HYBRID
    SampleT<Vector3>    SampleUniformDirection(const Vector2& xi);
    MRAY_HYBRID
    constexpr Float     PDFUniformDirection();
    //
    MRAY_HYBRID
    constexpr Optional<Spectrum> RussianRoulette(Spectrum, Float probability,
                                                 Float xi);
}

namespace Distribution::MIS
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

namespace Distribution
{

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::FresnelDielectric(Float cosFront, Float etaFront, Float etaBack)
{
    // Calculate Sin from Snell's Law
    Float sinFront = Math::SqrtMax(Float(1) - cosFront * cosFront);
    Float sinBack = etaFront / etaBack * sinFront;

    // Total internal reflection
    if(sinFront >= Float(1)) return Float(1);

    // Fresnel Equation
    Float cosBack = Math::SqrtMax(Float(1) - sinBack * sinBack);

    Float parallel = ((etaBack * cosFront - etaFront * cosBack) /
                      (etaBack * cosFront + etaFront * cosBack));
    parallel = parallel * parallel;

    float perpendicular = ((etaFront * cosFront - etaBack * cosBack) /
                           (etaFront * cosFront + etaBack * cosBack));
    perpendicular = perpendicular * perpendicular;

    return (parallel + perpendicular) * Float(0.5);
}

template<VectorC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::FresnelConductor(Float cosTheta, const T& eta, const T& k)
{
    // https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#FrConductor
    //
    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    // Find sin from trigonometry
    Float cosTheta2 = cosTheta * cosTheta;
    Float sinTheta2 = Float(1) - cosTheta2;
    T eta2 = eta * eta;
    T k2 = k * k;
    //
    T diff = eta2 - k2 - T(sinTheta2);
    T a2b2 = T::SqrtMax(diff * diff + Float(4) * eta2 * k2);
    T a = T::SqrtMax(T(0.5) * (a2b2 + diff));
    //
    T sT1 = a2b2 + cosTheta2;
    T sT2 = a * (Float(2) * cosTheta);
    T rS = (sT1 - sT2) / (sT1 + sT2);
    //
    T pT1 = a2b2 * cosTheta2 + T(sinTheta2 * sinTheta2);
    T pT2 = sT2 * sinTheta2;
    T rP = rS * (pT1 - pT2) / (pT1 + pT2);

    return (rP + rS) * Float(0.5);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::DGGX(Float NdH, Float alpha)
{
    Float alpha2 = alpha * alpha;
    Float denom = NdH * NdH * (alpha2 - Float(1)) + Float(1);
    denom = denom * denom;
    denom *= MathConstants::Pi<Float>();
    Float result = (alpha2 / denom);
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::LambdaSmith(const Vector3& vec, Float alpha)
{
    Vector3 vSqr = vec * vec;
    Float alpha2 = alpha * alpha;
    Float inner = alpha2 * (vSqr[0] + vSqr[1]) / vSqr[2];
    Float lambda = std::sqrt(Float(1) + inner) - Float(1);
    lambda *= Float(0.5);
    return lambda;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::GSmithSingle(const Vector3& vec, Float alpha)
{
    return Float(1) / (Float(1) + LambdaSmith(vec, alpha));

}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::GSmithCorrelated(const Vector3& wO, const Vector3& wI,
                             Float alpha)
{
    return Float(1) / (LambdaSmith(wO, alpha) + LambdaSmith(wI, alpha) + Float(1));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::GSmithSeparable(const Vector3& wO, const Vector3& wI,
                            Float alpha)
{
    return GSmithSingle(wO, alpha) * GSmithSingle(wI, alpha);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::GSchlick(Float cosTheta, Float alpha)
{
    if(cosTheta == Float(0)) return Float(0);
    Float k = alpha * Float(0.5);
    return cosTheta / (cosTheta * (1 - k) + k);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::GeomGGX(Float NdV, Float alpha)
{
    // Straight from paper (Eq. 34)
    // https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    if(NdV == Float(0)) return Float(0);

    // Eq has tan^2 theta which is
    // sin^2 / cos^2
    // sin^2 = 1 - cos^2
    Float cosTheta2 = NdV * NdV;
    Float tan2 = (Float(1) - cosTheta2) / cosTheta2;
    Float alpha2 = alpha * alpha;
    Float denom = Float(1) + std::sqrt(Float(1) + tan2 * alpha2);
    return Float(2) / denom;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum BxDF::FSchlick(Float VdH, const Spectrum& f0)
{
    // Classic Schlick's approx of fresnel term
    Float pw = Float(1) - VdH;
    Float pw2 = pw * pw;
    Float pw5 = pw2 * pw2 * pw;

    Spectrum result = (Spectrum(1) - f0) * pw5;
    result += f0;
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::VNDFGGXSmithPDF(const Vector3& V, const Vector3& H, Float alpha)
{
    Float VdH = std::max(Float(0), H.Dot(V));
    Float NdH = std::max(Float(0), H[2]);
    Float NdV = std::max(Float(0), V[2]);
    Float D = DGGX(NdH, alpha);
    Float GSingle = GSmithSingle(V, alpha);
    //
    if(NdV == Float(0)) return Float(0);
    //
    return VdH * D * GSingle / NdV;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> BxDF::VNDFGGXSmithSample(const Vector3& V, Float alpha,
                                          const Vector2& xi)
{
    // VNDF Routine straight from the paper
    // https://jcgt.org/published/0007/04/01/
    // G1 is Smith here be careful,
    // Everything is tangent space,
    // So no surface normal is feed to the system,
    // some Dot products (with normal) are thusly represented as
    // X[2] where x is the vector is being dot product with the normal
    //
    // Unlike most of the routines this sampling function
    // consists of multiple functions (namely NDF and Shadowing)
    // because of that, it does not return the value of the function
    // it returns the generated micro-facet normal
    //
    // And finally this routine represents isotropic material
    // a_y ==  a_x == a
    // Rename alpha for easier reading
    Float a = alpha;
    // Section 3.2 Ellipsoid to Spherical
    Vector3 VHemi = Vector3(a * V[0], a * V[1], V[2]).Normalize();
    // Section 4.1 Find orthonormal basis in the sphere
    Float len2 = Vector2f(VHemi).LengthSqr();
    Vector3 T1 = (len2 > 0)
                    ? Vector3(-VHemi[1], VHemi[0], 0.0f) / std::sqrt(len2)
                    : Vector3(1, 0, 0);
    Vector3 T2 = Vector3::Cross(VHemi, T1);
    // Section 4.2 Sampling using projected area
    Float r = std::sqrt(xi[0]);
    Float phi = Float(2) * MathConstants::Pi<Float>() * xi[1];
    Float t1 = r * std::cos(phi);
    Float t2 = r * std::sin(phi);
    Float s = Float(0.5) * (Float(1) + VHemi[2]);
    t2 = (Float(1) - s) * std::sqrt(Float(1) - t1 * t1) + s * t2;
    // Section 4.3: Projection onto hemisphere
    float val = Float(1) - t1 * t1 - t2 * t2;
    Vector3 NHemi = t1 * T1 + t2 * T2 + Math::SqrtMax(val) * VHemi;
    // Section 3.4: Finally back to Ellipsoid
    Vector3 NMicrofacet = Vector3(a * NHemi[0], a * NHemi[1],
                                  Math::SqrtMax(NHemi[2]));
    Float nLen2 = NMicrofacet.LengthSqr();
    if(nLen2 < MathConstants::Epsilon<Float>())
        NMicrofacet = Vector3::ZAxis();
    else
        NMicrofacet *= (Float(1) / std::sqrt(nLen2));

    return SampleT<Vector3>
    {
        .value = NMicrofacet,
        .pdf = VNDFGGXSmithPDF(V, NMicrofacet, alpha)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BxDF::BurleyDiffuseCorrection(Float NdL, Float NdV, Float LdH, Float roughness)
{
    // https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
    Float Fd90 = Float(0.5) + Float(2) * roughness * LdH * LdH;
    auto F = [&Fd90](Float dot)
    {
        Float pw = Float(1) - dot;
        Float pw2 = pw * pw;
        Float pw5 = pw2 * pw2 * pw;
        return Float(1) + (Fd90 - Float(1)) * pw5;
    };
    return F(NdL) * F(NdV);
}

template<VectorOrFloatC  T>
MRAY_HYBRID MRAY_CGPU_INLINE
T Common::DivideByPDF(T t, Float pdf)
{
    assert(pdf >= Float(0));
    if(pdf == Float(0)) return T(0);

    if constexpr(VectorC<T>)
    {
        Float pdfRecip = Float(1) / pdf;
        return t * pdfRecip;
    }
    else return t / pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::DotN(Vector3 v)
{
    return v[2];
}

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> Common::BisectSample1(Float xi, Float)
{
    return {0, xi};
}

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> Common::BisectSample2(Float xi, Vector2 weights,
                                            bool isAlreadyNorm)
{
    using Math::PrevFloat;
    if(!isAlreadyNorm) weights[0] /= weights.Sum();
    //
    Float w = weights[0];
    uint32_t i = (xi < w) ? 0 : 1;
    Float localXi = (xi < w)
            ? xi / w
            : (xi - w) / (Float(1) - w);
    localXi = std::min(localXi, PrevFloat<Float>(1));
    assert(localXi >= 0 && localXi < 1);
    return Pair<uint32_t, Float>(i, localXi);
}

template<uint32_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
Pair<uint32_t, Float> Common::BisectSample(Float xi, const Span<Float, N>& weights,
                                           bool isAlreadyNorm)
{
    using Math::PrevFloat;
    auto Reduce = [weights]() -> Float
    {
        Float r = 0;
        MRAY_UNROLL_LOOP
        for(uint32_t i = 1; i < N; i++)
            r += weights[i];
        return r;
    };
    auto Find = [weights](Float xiScaled) -> uint32_t
    {
        Float sum = 0;
        MRAY_UNROLL_LOOP
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
    Float diff = Float(0);
    if(i > 0) diff += weights[0];
    if(i > 1) diff += weights[1];
    Float localXi = (xiScaled - diff) / weights[i];
    localXi = (isAlreadyNorm) ? localXi : localXi * sum;
    localXi = std::min(localXi, PrevFloat<Float>(1));
    assert(localXi >= 0 && localXi < 1);
    return Pair<uint32_t, Float>(i, localXi);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleGaussian(Float xi, Float sigma, Float mu)
{
    Float x = MathConstants::Sqrt2<Float>() * sigma;
    Float e = Math::InvErrFunc(Float(2) * xi - Float(1));
    x = x * e + mu;

    // Erf can be -+inf, when xi is near zero or one
    // Just clamp to %99.95 estimate of the actual integral
    if(Math::IsInf(e))
    {
        Float minMax = Float(3.5) * sigma;
        x = Math::Clamp(x, -minMax, minMax);
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
    return Math::Gaussian(x, sigma, mu);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector2> Common::SampleGaussian2D(Vector2 xi, Float sigma,
                                          Vector2 mu)
{
    using namespace MathConstants;
    using Math::SinCos;
    // Instead of doing two gauss inverse sampling,
    // doing Box-Muller transform
    Float scalar = std::sqrt(Float(-2) * std::log(xi[0]));
    auto[s, c] = SinCos(Pi<Float>() * Float(2) * xi[1]);

    // Since rng is [0, 1) it can get zero then above function
    // If scalar is inf, we are at outer ring (infinitely long)
    // clamp similar to %99.5 of the range
    if(Math::IsInf(scalar)) scalar = Float(3.5);

    Vector2 xy = Vector2(scalar * s, scalar * c);
    xy =  (xy * sigma) + mu;
    Float pdf = PDFGaussian2D(xy, sigma, mu);
    return { xy, pdf };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFGaussian2D(Vector2 xy, Float sigma,
                            Vector2 mu)
{

    return (Math::Gaussian(xy[0], sigma, mu[0]) *
            Math::Gaussian(xy[1], sigma, mu[1]));
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleLine(Float xi, Float c, Float d)
{
    using namespace Math;
    // https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Sampling_Using_the_Inversion_Method#SampleLinear
    Float normVal = Float(2) / (c + d);
    // Avoid divide by zero
    const Float epsilon = NextFloat<Float>(0);
    if(c == 0 && xi == 0)
        return SampleT{epsilon, normVal * epsilon};

    Float denom = Lerp(c * c, d * d, xi);
    denom = c + std::sqrt(denom);
    Float x = (c + d) * xi / denom;
    using Math::PrevFloat;
    return SampleT<Float>
    {
        .value = std::min(x, PrevFloat<Float>(1)),
        .pdf = normVal * Lerp(c, d, x)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFLine(Float x, Float c, Float d)
{
    if(x < 0 && x > 1) return Float(0);
    Float normVal = Float(2) / (c + d);
    return normVal * Math::Lerp(c, d, x);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> Common::SampleTent(Float xi, Float a, Float b)
{
    // Dirac delta like, return as if dirac delta
    if(b - a < MathConstants::LargeEpsilon<Float>())
        return {Float(0), Float(1) / (b - a)};

    using Math::PrevFloat;
    assert(a <= 0 && b >= 0);
    auto [index, localXi] = BisectSample2(xi, Vector2(-a, b), false);
    localXi = (index == 0) ? (PrevFloat<Float>(1) - localXi) : localXi;
    SampleT<Float> result = SampleLine(localXi, 1, 0);
    Float& x = result.value;
    x = (index == 0) ? (x * a) : (x * b);
    return SampleT<Float>
    {
        result.value,
        result.pdf * Float(1) / (b - a)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Common::PDFTent(Float x, Float a, Float b)
{
    assert(a <= x && b >= x);
    if((b - a) < MathConstants::LargeEpsilon<Float>())
        return Float(1) / (b - a);

    //Float mid = a + (b - a) * Float(0.5);
    Float x01 = std::abs(1 - x);
    x01 = (x < 0) ? (x / a) : (x / b);
    //Float pdf = (x < 0) ? (-a) : b;
    Float pdf = Float(1) / (b - a);
    pdf *= PDFLine(x01, 1, 0);
    return pdf;
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
    return Float(1) / (b - a);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> Common::SampleCosDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using Math::SqrtMax;

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
    using namespace MathConstants;
    Float pdf = n.Dot(v) * InvPi<Float>();
    pdf = (pdf <= Epsilon<Float>()) ? Float(0) : pdf;
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Float Common::PDFCosDirection(const Vector3& v)
{
    using namespace MathConstants;
    Float pdf = v[2] * InvPi<Float>();
    pdf = (pdf <= Epsilon<Float>()) ? Float(0) : pdf;
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> Common::SampleUniformDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using Math::SqrtMax;

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

//
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Optional<Spectrum> Common::RussianRoulette(Spectrum throughput,
                                                     Float probability, Float xi)
{
    // We clamp the probability here.
    // If prob is too low, fireflies become too large (value wise)
    probability = Math::Clamp(probability, Float(0.1), Float(1));
    if(xi >= probability)
        return std::nullopt;
    else
        return throughput * Float(1) / probability;
}

template<uint32_t N>
MRAY_HYBRID
Float MIS::BalanceCancelled(const Span<Float, N>& pdfs,
                            const Span<Float, N>& weights)
{
    Float result = Float(0);
    MRAY_UNROLL_LOOP
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

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Spectrum Medium::WavesToSpectrumCauchy(const SpectrumWaves& waves,
                                                 const Vector3& coeffs)
{
    Spectrum result;
    MRAY_UNROLL_LOOP
    for(uint32_t i = 0; i < SpectrumWaves::Dims; i++)
    {
        Float w2 = waves[i] * waves[i];
        Float w4 = w2 * w2;
        result[i] = coeffs[0] + coeffs[1] / w2 + coeffs[2] / w4;
    }
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Medium::HenyeyGreensteinPhase(Float cosTheta, Float g)
{
    // From the PBR book
    // https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions#HenyeyGreenstein

    Float gSqr = g * g;
    Float nom = (Float(1) - gSqr) * MathConstants::Inv4Pi<Float>();
    // https://www.desmos.com/calculator/7ogsbedc2r
    // the sqrtIn is never technically zero
    // due to numerical errors it can be zero
    Float sqrtIn = Float(1) + gSqr + cosTheta;
    Float denom = Math::SqrtMax(sqrtIn);
    denom *= sqrtIn;
    return nom / denom;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> Medium::SampleHenyeyGreensteinPhase(const Vector3& wO, Float g,
                                                     const Vector2& xi)
{
    using namespace Math;
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
    Vector3 wI = Graphics::UnitSphericalToCartesian(sinCosPhi, sinCosTheta);
    Quaternion rot = Quaternion::RotationBetweenZAxis(wO);
    wI = rot.ApplyRotation(wI);

    return SampleT<Vector3>
    {
        .value = wI,
        .pdf = HenyeyGreensteinPhase(cosTheta, g)
    };
}


}