#pragma once

#include "Core/Vector.h"
#include "Core/GraphicsFunctions.h"
#include "Core/Math.h"
#include "TracerTypes.h"

namespace Distribution
{
    // TODO: This design was going to be used for vMF
    // storage for vMF-based path guiding.
    //
    // But I implemented Alber2025
    // https://dl.acm.org/doi/10.1145/3728296
    // That paper stores sufficient statistics instead of actual kernel.
    // so this is shelved. It is not deleted since comments may be
    // useful later.
    namespace Shelved
    {
        // A gaussian lobe (aka. vMF distribution) in two dimension.
        // with anisotrophy (we can fit 1 word here)
        struct GaussLobeDirParams
        {
            SNorm2x16   dirCoOcta; // CoOcta coordinates of the normal vector
                                   // over unit sphere
        };

        struct GaussLobeMeanParams
        {
            // This min max is visually inspected via desmos
            // Unfortunately did not saved the graph so can't share it :(
            static constexpr uint32_t   KAPPA_MAX = 1024;
            static constexpr Float      KAPPA_DELTA = Float(KAPPA_MAX) / UNorm2x16::Max();
            // [0, 1024] UNorm16 value;
            uint16_t    kappa;
        };

        // TODO: We may further compress this in log scale maybe
        // since these will be pdf most of the time.
        struct GaussLobeScaleParams
        {
            // Single channel data only overall Luminance maybe (for PG)
            // depending on the situation, this struct may be multi
            // channle or multi-spectra. (Unlikely since this means extra memory
            Float scale;
        };
        // TODO: Add anisotrophy to the system.
        // Multiplying aniso with another aniso distribution
        // is not straightforward. So it is shelved for now.
        struct GaussLobeAnisoParams
        {
            // Disney's principaled BxDF style aniso
            // (at least this is inspired from it).
            //
            // "beta" is Kent Distribution "beta" (Kent Distribution is
            // Aniso of GaussLobe aka. vMF)
            // Constraint:
            //   0 <= "beta" < UNorm * (kappa / 2)
            //
            // Rotation of the aniso disk between [0-180) (in radians)
            // Aniso basis will be deterministic
            // Not final but we rotate x_y basis with this rotation than
            // align Z with the "dir" param.
            UNorm2x16   betaAndRotation;
        };
        static constexpr auto DIR_INDEX   = 0;
        static constexpr auto MEAN_INDEX  = 1;
        static constexpr auto SCALE_INDEX = 2;
        using GaussLobeSoA = SoASpan<GaussLobeDirParams,
                                     GaussLobeMeanParams,
                                     GaussLobeScaleParams>;
        using LobeMixtureSizeT = uint8_t;
    }

    struct GaussianLobe
    {
        Vector3 dir;
        Float   kappa;
        Float   alpha;

        // This is for numeric stability
        static constexpr Float MIN_K = MathConstants::LargeEpsilon<Float>();
        static constexpr Float MAX_K = Float(1) / MIN_K;

        // Helpers
        MR_HF_DECL static Float XOverExp1M(Float x);
        MR_HF_DECL static Float XOverExp1MRecip(Float x);

        constexpr  GaussianLobe() = default;
        MR_HF_DECL GaussianLobe(const Vector3& dir, Float kappa);
        MR_HF_DECL GaussianLobe(const Vector3& dir, Float kappa, Float alpha);

        MR_HF_DECL SampleT<Vector3> Sample(Vector2 xi) const;
        MR_HF_DECL Float            Value(Vector3 wI) const;
        MR_HF_DECL Float            Pdf(Vector3 wI) const;
        MR_HF_DECL Float            SolidAnglePDF(Vector3 wI) const;
        MR_HF_DECL GaussianLobe     Convolution(const GaussianLobe& other) const;
        MR_HF_DECL GaussianLobe     Product(const GaussianLobe& other) const;
    };

}

namespace Distribution::BxDF
{
    MR_PF_DECL Float FresnelDielectric(Float cosFront, Float etaFront, Float etaBack);
    template<VectorC T>
    MR_PF_DECL Float FresnelConductor(Float cosFront, const T& etaFront, const T& etaBack);
    MR_PF_DECL Float DGGX(Float NdH, Float alpha);

    MR_PF_DECL Float LambdaSmith(const Vector3& vec, Float alpha);

    MR_PF_DECL Float GSmithSingle(const Vector3& vec, Float alpha);
    MR_PF_DECL Float GSmithCorrelated(const Vector3& wO, const Vector3& wI, Float alpha);
    MR_PF_DECL Float GSmithSeparable(const Vector3& wO, const Vector3& wI, Float alpha);
    MR_PF_DECL Float GSchlick(Float cosTheta, Float alpha);
    MR_PF_DECL Float GeomGGX(Float cosTheta, Float alpha);

    MR_PF_DECL Spectrum         FSchlick(Float VdH, const Spectrum& f0);
    MR_PF_DECL Float            VNDFGGXSmithPDF(const Vector3& V, const Vector3& H, Float alpha);
    MR_PF_DECL SampleT<Vector3> VNDFGGXSmithSample(const Vector3& V, Float alpha, const Vector2& xi);
    MR_PF_DECL Float            BurleyDiffuseCorrection(Float NdL, Float NdV, Float LdH, Float roughness);
}

namespace Distribution::Medium
{
    MR_PF_DECL
    Float               WavelengthToIoRCauchy(Float wavelength,
                                              const Vector3& coeffs);
    MR_PF_DECL
    Float               HenyeyGreensteinPhase(Float cosTheta, Float g);

    MR_HF_DECL
    SampleT<Vector3>    SampleHenyeyGreensteinPhase(const Vector3& wO, Float g,
                                                    const Vector2& xi);
}

namespace Distribution::Common
{
    template<class T>
    concept FloatOrVecFloatC = FloatVectorC<T> || FloatC<T>;

    template<FloatOrVecFloatC  T>
    MR_PF_DECL T       DivideByPDF(T, Float pdf);
    MR_PF_DECL Float   DotN(Vector3);

    MR_PF_DECL
    Pair<uint32_t, Float>   BisectSample1(Float xi, Float weight);
    MR_PF_DECL
    Pair<uint32_t, Float>   BisectSample2(Float xi, Vector2 weights,
                                          bool isAlreadyNorm = false);
    template<uint32_t N>
    MR_PF_DECL
    Pair<uint32_t, Float>   BisectSample(Float xi, const Span<Float, N>& weights,
                                         bool isAlreadyNorm = false);

    MR_PF_DECL
    SampleT<Float>  SampleGaussian(Float xi, Float sigma = Float(1),
                                   Float mu = Float(0));
    MR_PF_DECL
    Float           PDFGaussian(Float x, Float sigma = Float(1),
                                Float mu = Float(0));

    // TODO: Only providing isotropic version
    // here, anisotropic version may not be useful
    // We may provide a spherical version later.
    MR_PF_DECL
    SampleT<Vector2>    SampleGaussian2D(Vector2 xi, Float sigma = Float(1),
                                         Vector2 mu = Vector2::Zero());
    MR_PF_DECL
    Float               PDFGaussian2D(Vector2 xy, Float sigma = Float(1),
                                      Vector2 mu = Vector2::Zero());

    MR_PF_DECL
    SampleT<Float>  SampleLine(Float xi, Float c, Float d);
    MR_PF_DECL
    Float           PDFLine(Float x, Float c, Float d);

    MR_PF_DECL
    SampleT<Float>  SampleTent(Float xi, Float a, Float b);
    MR_PF_DECL
    Float           PDFTent(Float x, Float a, Float b);

    MR_PF_DECL
    SampleT<Float>  SampleUniformRange(Float xi, Float a, Float b);
    MR_PF_DECL
    Float           PDFUniformRange(Float x, Float a, Float b);
    //
    MR_PF_DECL
    SampleT<Vector3>    SampleCosDirection(const Vector2& xi);
    MR_PF_DECL
    Float               PDFCosDirection(const Vector3& v, const Vector3& n);
    MR_PF_DECL
    Float               PDFCosDirection(const Vector3& v);
    MR_PF_DECL
    SampleT<Vector3>    SampleUniformDirection(const Vector2& xi);
    MR_PF_DECL
    Float               PDFUniformDirection();
    //
    MR_PF_DECL
    Optional<Spectrum>  RussianRoulette(Spectrum, Float probability, Float xi);
}

namespace Distribution::MIS
{
    template<uint32_t N>
    MR_PF_DECL
    Float BalanceCancelled(const Span<const Float, N>& pdfs,
                           const Span<const Float, N>& weights);

    template<uint32_t N>
    MR_PF_DECL
    Float Balance(uint32_t pdfIndex,
                  const Span<const Float, N>& pdfs,
                  const Span<const Float, N>& weights);
}

namespace Distribution
{

MR_HF_DEF
GaussianLobe::GaussianLobe(const Vector3& dir, Float kappa)
    : dir(dir)
    , kappa(kappa)
    , alpha(XOverExp1M(Float(-2) * kappa))
{}

MR_HF_DEF
GaussianLobe::GaussianLobe(const Vector3& dir, Float kappa, Float alpha)
    : dir(dir)
    , kappa(kappa)
    , alpha(alpha)
{}

MR_HF_DEF
SampleT<Vector3>
GaussianLobe::Sample(Vector2 xi) const
{
    // Numerically stable vMF sampling
    // https://gpuopen.com/download/A_Numerically_Stable_Implementation_of_the_von_Mises%E2%80%93Fisher_Distribution_on_S2.pdf
    static constexpr auto MachineEpsilon = std::numeric_limits<Float>::epsilon();
    static constexpr auto T = MachineEpsilon / Float(4);

    Float r = 0;
    if(kappa > T)
    {
        r = (Float(1) / kappa);
        r *= Math::Log1P(xi[1] * Math::ExpM1(Float(-2) * kappa));
    }
    else r = Float(-2) * xi[1];

    Float cosPhi = 1 + r;
    Float sinPhi = Math::Sqrt(-Math::FMA(r, r, Float(2) * r));
    Float theta = Float(2) * MathConstants::Pi<Float>() * xi[0];
    auto [sinTheta, cosTheta] = Math::SinCos(theta);
    Vector3 dirZ = Graphics::UnitSphericalToCartesian(Vector2(sinTheta, cosTheta),
                                                        Vector2(sinPhi, cosPhi));
    // Do orientation
    Quaternion rot = Quaternion::RotationBetweenZAxis(dirZ);
    Vector3 dirWorld = rot.ApplyRotation(dir);
    //
    return SampleT<Vector3>
    {
        .value = dirWorld,
        .pdf = Pdf(dirWorld)
    };
}

MR_HF_DEF
Float
GaussianLobe::XOverExp1M(Float x)
{
    Float u = Math::Exp(x);
    //
    if(u == Float(1))           return Float(1);
    if(Math::Abs(x) < Float(1)) return Math::Log(u) / (u - Float(1));
    else                        return x            / (u - Float(1));
};

MR_HF_DEF
Float
GaussianLobe::XOverExp1MRecip(Float x)
{
    Float u = Math::Exp(x);
    //
    if(u == Float(1))           return Float(1);
    if(Math::Abs(x) < Float(1)) return (u - Float(1)) / Math::Log(u);
    else                        return (u - Float(1)) / x;
};

MR_HF_DEF
Float
GaussianLobe::Value(Vector3 wO) const
{
    Vector3 d = dir - wO;
    Float result = Math::Exp(Float(-0.5) * kappa * Math::LengthSqr(d));
    result *= alpha;
    result *= MathConstants::Inv4Pi<Float>();
    return result;
}

MR_HF_DEF
Float
GaussianLobe::Pdf(Vector3 wO) const
{
    return Value(wO);
}

MR_HF_DEF
Float
GaussianLobe::SolidAnglePDF(Vector3 wI) const
{
    // TODO: How to convert to solid angle density???
    // or is it already solid angle
    return Pdf(wI);
}

MR_HF_DEF
GaussianLobe
GaussianLobe::Convolution(const GaussianLobe& other) const
{
    // https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    auto A3 = [](Float k)
    {
        Float r = Float(1) / Math::TanH(k);
        r -= Float(1) / k;
        return r;
    };
    auto dA3 = [](Float k)
    {
        Float c = Float(2) / (Math::Exp(k) - Math::Exp(-k));
        Float r = Float(1) / (k * k) - (c * c);
        return r;
    };

    // Newton-Raphson
    Float x = Math::Min(kappa, other.kappa);
    Float y = A3(kappa) * A3(other.kappa);
    Float residual = 0;
    do
    {
        residual = A3(x) - y;
        x -= residual / dA3(x);
    }
    while(Math::Abs(residual) > MathConstants::SmallEpsilon<Float>());

    //
    return GaussianLobe(dir, x, XOverExp1M(Float(-2) * x));
}

MR_HF_DEF
GaussianLobe
GaussianLobe::Product(const GaussianLobe& other) const
{
    Float k1 = kappa;
    Float k2 = other.kappa;
    Vector3 d1 = dir;
    Vector3 d2 = other.dir;

    Vector3 newD = d1 * k1 + d2 * k2;
    Float length = Math::Length(newD);
    Float newK = Math::Clamp(length, MIN_K, MAX_K);
    Float newAlpha = XOverExp1M(Float(-2) * k1) * XOverExp1M(Float(-2) * k2);
    newAlpha *= XOverExp1MRecip(Float(-2) * newK);

    newD /= length;
    return GaussianLobe(newD, newK, newAlpha);
}

MR_PF_DEF
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
MR_PF_DEF
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
    T a2b2 = Math::SqrtMax(diff * diff + Float(4) * eta2 * k2);
    T a = Math::SqrtMax(T(0.5) * (a2b2 + diff));
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

MR_PF_DEF
Float BxDF::DGGX(Float NdH, Float alpha)
{
    Float alpha2 = alpha * alpha;
    Float denom = NdH * NdH * (alpha2 - Float(1)) + Float(1);
    denom = denom * denom;
    denom *= MathConstants::Pi<Float>();
    Float result = (alpha2 / denom);
    return result;
}

MR_PF_DEF
Float BxDF::LambdaSmith(const Vector3& vec, Float alpha)
{
    Vector3 vSqr = vec * vec;
    Float alpha2 = alpha * alpha;
    Float inner = alpha2 * (vSqr[0] + vSqr[1]) / vSqr[2];
    Float lambda = Math::Sqrt(Float(1) + inner) - Float(1);
    lambda *= Float(0.5);
    return lambda;
}

MR_PF_DEF
Float BxDF::GSmithSingle(const Vector3& vec, Float alpha)
{
    return Float(1) / (Float(1) + LambdaSmith(vec, alpha));

}

MR_PF_DEF
Float BxDF::GSmithCorrelated(const Vector3& wO, const Vector3& wI,
                             Float alpha)
{
    return Float(1) / (LambdaSmith(wO, alpha) + LambdaSmith(wI, alpha) + Float(1));
}

MR_PF_DEF
Float BxDF::GSmithSeparable(const Vector3& wO, const Vector3& wI,
                            Float alpha)
{
    return GSmithSingle(wO, alpha) * GSmithSingle(wI, alpha);
}

MR_PF_DEF
Float BxDF::GSchlick(Float cosTheta, Float alpha)
{
    if(cosTheta == Float(0)) return Float(0);
    Float k = alpha * Float(0.5);
    return cosTheta / (cosTheta * (1 - k) + k);
}

MR_PF_DEF
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
    Float denom = Float(1) + Math::Sqrt(Float(1) + tan2 * alpha2);
    return Float(2) / denom;
}

MR_PF_DEF
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

MR_PF_DEF
Float BxDF::VNDFGGXSmithPDF(const Vector3& V, const Vector3& H, Float alpha)
{
    Float VdH = Math::Max(Float(0), Math::Dot(H, V));
    Float NdH = Math::Max(Float(0), H[2]);
    Float NdV = Math::Max(Float(0), V[2]);
    Float D = DGGX(NdH, alpha);
    Float GSingle = GSmithSingle(V, alpha);
    //
    if(NdV == Float(0)) return Float(0);
    //
    return VdH * D * GSingle / NdV;
}

MR_PF_DEF
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
    // Unlike most of the routines, this sampling function
    // consists of multiple functions (namely NDF and Shadowing)
    // because of that, it does not return the value of the function
    // it returns the generated micro-facet normal
    //
    // And finally this routine represents isotropic material
    // a_y ==  a_x == a
    // Rename alpha for easier reading
    Float a = alpha;
    // Section 3.2 Ellipsoid to Spherical
    Vector3 VHemi = Math::Normalize(Vector3(a * V[0], a * V[1], V[2]));
    // Section 4.1 Find orthonormal basis in the sphere
    Float len2 = Math::LengthSqr(Vector2(VHemi));
    Vector3 T1 = (len2 > 0)
                    ? Vector3(-VHemi[1], VHemi[0], 0.0f) * Math::RSqrt(len2)
                    : Vector3(1, 0, 0);
    Vector3 T2 = Math::Cross(VHemi, T1);
    // Section 4.2 Sampling using projected area
    Float r = Math::Sqrt(xi[0]);
    Float phi = Float(2) * MathConstants::Pi<Float>() * xi[1];
    const auto& [sinPhi, cosPhi] = Math::SinCos(phi);
    Float t1 = r * cosPhi;
    Float t2 = r * sinPhi;
    Float s = Float(0.5) * (Float(1) + VHemi[2]);
    t2 = (Float(1) - s) * Math::Sqrt(Float(1) - t1 * t1) + s * t2;
    // Section 4.3: Projection onto hemisphere
    float val = Float(1) - t1 * t1 - t2 * t2;
    Vector3 NHemi = t1 * T1 + t2 * T2 + Math::SqrtMax(val) * VHemi;
    // Section 3.4: Finally back to Ellipsoid
    Vector3 NMicrofacet = Vector3(a * NHemi[0], a * NHemi[1],
                                  Math::SqrtMax(NHemi[2]));
    Float nLen2 = Math::LengthSqr(NMicrofacet);
    if(nLen2 < MathConstants::Epsilon<Float>())
        NMicrofacet = Vector3::ZAxis();
    else
        NMicrofacet *= Math::RSqrt(nLen2);

    return SampleT<Vector3>
    {
        .value = NMicrofacet,
        .pdf = VNDFGGXSmithPDF(V, NMicrofacet, alpha)
    };
}

MR_PF_DEF
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

template<Common::FloatOrVecFloatC  T>
MR_PF_DEF
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

MR_PF_DEF
Float Common::DotN(Vector3 v)
{
    return v[2];
}

MR_PF_DEF
Pair<uint32_t, Float> Common::BisectSample1(Float xi, Float)
{
    return {0, xi};
}

MR_PF_DEF
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
    localXi = Math::Min(localXi, PrevFloat<Float>(1));
    assert(localXi >= 0 && localXi < 1);
    return Pair<uint32_t, Float>(i, localXi);
}

template<uint32_t N>
MR_PF_DEF
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
    localXi = Math::Min(localXi, PrevFloat<Float>(1));
    assert(localXi >= 0 && localXi < 1);
    return Pair<uint32_t, Float>(i, localXi);
}

MR_PF_DEF
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

MR_PF_DEF
Float Common::PDFGaussian(Float x, Float sigma, Float mu)
{
    return Math::Gaussian(x, sigma, mu);
}

MR_PF_DEF
SampleT<Vector2> Common::SampleGaussian2D(Vector2 xi, Float sigma,
                                          Vector2 mu)
{
    using namespace MathConstants;
    // Instead of doing two gauss inverse sampling,
    // doing Box-Muller transform
    Float scalar = Math::Sqrt(Float(-2) * Math::Log(xi[0]));
    auto [s, c] = Math::SinCos(Pi<Float>() * Float(2) * xi[1]);

    // Since rng is [0, 1) it can get zero then above function
    // If scalar is inf, we are at outer ring (infinitely long)
    // clamp similar to %99.5 of the range
    if(Math::IsInf(scalar)) scalar = Float(3.5);

    Vector2 xy = Vector2(scalar * s, scalar * c);
    xy =  (xy * sigma) + mu;
    Float pdf = PDFGaussian2D(xy, sigma, mu);
    return { xy, pdf };
}

MR_PF_DEF
Float Common::PDFGaussian2D(Vector2 xy, Float sigma,
                            Vector2 mu)
{

    return (Math::Gaussian(xy[0], sigma, mu[0]) *
            Math::Gaussian(xy[1], sigma, mu[1]));
}

MR_PF_DEF
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
    denom = c + Math::Sqrt(denom);
    Float x = (c + d) * xi / denom;
    using Math::PrevFloat;
    return SampleT<Float>
    {
        .value = Math::Min(x, PrevFloat<Float>(1)),
        .pdf = normVal * Lerp(c, d, x)
    };
}

MR_PF_DEF
Float Common::PDFLine(Float x, Float c, Float d)
{
    if(x < 0 && x > 1) return Float(0);
    Float normVal = Float(2) / (c + d);
    return normVal * Math::Lerp(c, d, x);
}

MR_PF_DEF
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

MR_PF_DEF
Float Common::PDFTent(Float x, Float a, Float b)
{
    assert(a <= x && b >= x);
    if((b - a) < MathConstants::LargeEpsilon<Float>())
        return Float(1) / (b - a);

    //Float mid = a + (b - a) * Float(0.5);
    Float x01 = Math::Abs(1 - x);
    x01 = (x < 0) ? (x / a) : (x / b);
    //Float pdf = (x < 0) ? (-a) : b;
    Float pdf = Float(1) / (b - a);
    pdf *= PDFLine(x01, 1, 0);
    return pdf;
}

MR_PF_DEF
SampleT<Float> Common::SampleUniformRange(Float xi, Float a, Float b)
{
    return SampleT<Float>
    {
        .value = xi * (b - a) + a,
        .pdf = Float(1) / (b - a)
    };
}

MR_PF_DEF
Float Common::PDFUniformRange(Float x, Float a, Float b)
{
    if(x < a && x > b) return 0;
    return Float(1) / (b - a);
}

MR_PF_DEF
SampleT<Vector3> Common::SampleCosDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using Math::SqrtMax;

    // Generated direction is on unit space (+Z oriented hemisphere)
    Float xi1Angle = Float{2} * Pi<Float>() * xi[1];
    Float xi0Sqrt = Math::Sqrt(xi[0]);

    Vector3 dir;
    const auto& [s, c] = Math::SinCos(xi1Angle);
    dir[0] = xi0Sqrt * c;
    dir[1] = xi0Sqrt * s;
    dir[2] = SqrtMax(Float{1} - Math::LengthSqr(Vector2(dir)));

    // Fast tangent space dot product and domain constant
    Float pdf = dir[2] * InvPi<Float>();

    // Finally the result!
    return SampleT<Vector3>
    {
        .value = dir,
        .pdf = pdf
    };
}

MR_PF_DEF
Float Common::PDFCosDirection(const Vector3& v, const Vector3& n)
{
    using namespace MathConstants;
    Float pdf = Math::Dot(n, v) * InvPi<Float>();
    pdf = (pdf <= Epsilon<Float>()) ? Float(0) : pdf;
    return pdf;
}

MR_PF_DEF
Float Common::PDFCosDirection(const Vector3& v)
{
    using namespace MathConstants;
    Float pdf = v[2] * InvPi<Float>();
    pdf = (pdf <= Epsilon<Float>()) ? Float(0) : pdf;
    return pdf;
}

MR_PF_DEF
SampleT<Vector3> Common::SampleUniformDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using Math::SqrtMax;

    Float xi0Sqrt = SqrtMax(Float{1} - xi[0] * xi[0]);
    Float xi1Angle = 2 * Pi<Float>() * xi[1];

    const auto& [sinX1, cosX1] = Math::SinCos(xi1Angle);
    Vector3 dir;
    dir[0] = xi0Sqrt * cosX1;
    dir[1] = xi0Sqrt * sinX1;
    dir[2] = xi[0];

    // Uniform pdf is invariant
    constexpr Float pdf = InvPi<Float>() * Float{0.5};
    return SampleT<Vector3>
    {
        .value = dir,
        .pdf = pdf
    };
}

MR_PF_DEF
Float Common::PDFUniformDirection()
{
    return MathConstants::InvPi<Float>() * Float{0.5};
}

//
MR_PF_DEF
Optional<Spectrum> Common::RussianRoulette(Spectrum throughput,
                                           Float probability, Float xi)
{
    // We clamp the probability here.
    // If prob is too low, fireflies become too large (value wise)
    probability = Math::Clamp(probability, Float(0.1), Float(1));
    if(xi >= probability)
        return std::nullopt;
    else
        return throughput * (Float(1) / probability);
}

template<uint32_t N>
MR_PF_DEF
Float MIS::BalanceCancelled(const Span<const Float, N>& pdfs,
                            const Span<const Float, N>& weights)
{
    Float result = Float(0);
    MRAY_UNROLL_LOOP
    for(uint32_t i = 0; i < N; i++)
        result += pdfs[i] * weights[i];
    return result;
}

template<uint32_t N>
MR_PF_DEF
Float MIS::Balance(uint32_t pdfIndex,
                   const Span<const Float, N>& pdfs,
                   const Span<const Float, N>& weights)
{
    Float denom = BalanceCancelled(pdfIndex, pdfs, weights);
    return weights[pdfIndex] * pdfs[pdfIndex] / denom;
}

MR_PF_DEF
Float Medium::WavelengthToIoRCauchy(Float wavelength, const Vector3& coeffs)
{
    // Cauchy formulae requires wavelengths to be in micrometers
    Float w = wavelength * Float(1e-3);
    Float w2 = w * w;
    Float w4 = w2 * w2;
    Float ior = coeffs[0] + coeffs[1] / w2 + coeffs[2] / w4;
    return ior;
}

MR_PF_DEF
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

MR_HF_DEF
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
    bool isNearZero = (Abs(g) < MathConstants::VeryLargeEpsilon<Float>());
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