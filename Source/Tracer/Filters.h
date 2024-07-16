#pragma once

#include "Core/Vector.h"

#include "DistributionFunctions.h"

class BoxFilter
{
    private:
    Float   radius;

    public:
    // Constructors & Destructor
    MRAY_HOST           BoxFilter(Float radius);
    //
    MRAY_HYBRID
    Float               Evaluate(const Vector2& duv) const;
    MRAY_HYBRID
    SampleT<Vector2>    Sample(const Vector2& xi) const;
    MRAY_HYBRID
    Float               Pdf(const Vector2& duv) const;
};

class TentFilter
{
    private:
    Float radius;
    Float recipRadius;

    public:
    // Constructors & Destructor
    MRAY_HOST           TentFilter(Float radius);
    //
    MRAY_HYBRID
    Float               Evaluate(const Vector2& duv) const;
    MRAY_HYBRID
    SampleT<Vector2>    Sample(const Vector2& xi) const;
    MRAY_HYBRID
    Float               Pdf(const Vector2& duv) const;
};

class GaussianFilter
{
    private:
    Float radius;
    Float sigma;

    public:
    // Constructors & Destructor
    MRAY_HOST           GaussianFilter(Float radius);
    //
    MRAY_HYBRID
    Float               Evaluate(const Vector2& duv) const;
    MRAY_HYBRID
    SampleT<Vector2>    Sample(const Vector2& xi) const;
    MRAY_HYBRID
    Float               Pdf(const Vector2& duv) const;
};

class MitchellNetravaliFilter
{
    static constexpr auto MIS_MID       = Float(0.97619047619);
    static constexpr auto MIS_SIDES     = Float(0.0119047619048);
    static constexpr auto SIDE_MEAN     = Float(1.429);
    static constexpr auto SIDE_STD_DEV  = Float(0.4);
    static constexpr auto MID_STD_DEV   = Float(0.15);
    // Wow this works, compile time float comparison
    static_assert((MIS_MID + 2 * MIS_SIDES) == 1, "Weights are not normalized!");

    private:
    Float   radius;
    Float   radiusRecip;
    Vector4 coeffs01;
    Vector4 coeffs12;
    // Modified values via radius
    Float   midSigma;
    Float   sideSigma;
    Float   sideMean;

    public:
    // Constructors & Destructor
    MRAY_HOST           MitchellNetravaliFilter(Float radius,
                                                Float b = Float(0.33333),
                                                Float c = Float(0.33333));
    //
    MRAY_HYBRID
    Float               Evaluate(const Vector2& duv) const;
    MRAY_HYBRID
    SampleT<Vector2>    Sample(const Vector2& xi) const;
    MRAY_HYBRID
    Float               Pdf(const Vector2& duv) const;
};

MRAY_HOST inline
BoxFilter::BoxFilter(Float r)
    : radius(r)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BoxFilter::Evaluate(const Vector2& duv) const
{
    Vector2 t = duv.Abs();
    Float rr = Float(1) / radius;
    return (t[0] <= radius || t[1] <= radius) ? (Float(0.25) * rr * rr) : Float(0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector2> BoxFilter::Sample(const Vector2& xi) const
{
    using namespace Distribution;
    auto r0 = Common::SampleUniformRange(xi[0], -radius, radius);
    auto r1 = Common::SampleUniformRange(xi[1], -radius, radius);
    return SampleT<Vector2>
    {
        .value = Vector2(r0.value, r1.value),
        .pdf = r0.pdf * r1.pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float BoxFilter::Pdf(const Vector2& duv) const
{
    using namespace Distribution;
    return (Common::PDFUniformRange(duv[0], -radius, radius) *
            Common::PDFUniformRange(duv[1], -radius, radius));
}

MRAY_HOST inline
TentFilter::TentFilter(Float r)
    : radius(r)
    , recipRadius(1.0f / radius)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
Float TentFilter::Evaluate(const Vector2& duv) const
{
    using namespace MathFunctions;
    Vector2 t = (duv * recipRadius).Abs();
    Float capSqrt = Float(1) / radius;
    // TODO: This seems wrong? Pyramdi cap (height)
    // should be 3/(4 * r^2) ?
    Float cap = capSqrt;
    Float x = Lerp<Float>(cap, 0, t[0]);
    Float y = Lerp<Float>(cap, 0, t[1]);
    return x * y;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector2> TentFilter::Sample(const Vector2& xi) const
{
    using namespace Distribution;
    auto s0 = Common::SampleTent(xi[0], -radius, radius);
    auto s1 = Common::SampleTent(xi[1], -radius, radius);
    return SampleT<Vector2>
    {
        .value = Vector2(s0.value, s1.value),
        .pdf = s0.pdf * s1.pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float TentFilter::Pdf(const Vector2& duv) const
{
    using namespace Distribution;
    Float x = Common::PDFTent(duv[0], -radius, radius);
    Float y = Common::PDFTent(duv[0], -radius, radius);
    return x * y;
}

MRAY_HOST inline
GaussianFilter::GaussianFilter(Float r)
    : radius(r)
    // ~%95 of samples lies between [-r,r]
    , sigma(r * Float(0.5))
{}

MRAY_HYBRID MRAY_CGPU_INLINE
Float GaussianFilter::Evaluate(const Vector2& duv) const
{
    return (MathFunctions::Gaussian(duv[0], sigma) *
            MathFunctions::Gaussian(duv[1], sigma));
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector2> GaussianFilter::Sample(const Vector2& xi) const
{
    using namespace Distribution;
    auto r0 = Common::SampleGaussian(xi[0], sigma);
    auto r1 = Common::SampleGaussian(xi[1], sigma);
    return SampleT<Vector2>
    {
        .value = Vector2(r0.value,
                                 r1.value),
        .pdf = r0.pdf * r1.pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float GaussianFilter::Pdf(const Vector2& duv) const
{
    using namespace Distribution;
    return (Common::PDFGaussian(duv[0], sigma) *
            Common::PDFGaussian(duv[1], sigma));
}

MRAY_HOST inline
MitchellNetravaliFilter::MitchellNetravaliFilter(Float r, Float b, Float c)
    : radius(r)
    , radiusRecip(Float(1) / radius)
    , midSigma(MID_STD_DEV * radiusRecip * Float(2))
    , sideSigma(SIDE_STD_DEV * radiusRecip* Float(2))
    , sideMean(SIDE_MEAN * radiusRecip * Float(2))
{
    static constexpr Float F = Float(1) / Float(6);
    coeffs01[0] = F * (Float(12) - Float(9) * b - Float(6) * c);
    coeffs01[1] = F * (Float(-18) + Float(12) * b + Float(6) * c);
    coeffs01[2] = Float(0);
    coeffs01[3] = F * (Float(6) - Float(2) * b);

    coeffs12[0] = F * (-b - Float(6) * c);
    coeffs12[1] = F * (Float(6) * b + Float(30) * c);
    coeffs12[2] = F * (Float(-12) * b - Float(48) * c);
    coeffs12[3] = F * (Float(8) * b + Float(24) * c);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float MitchellNetravaliFilter::Evaluate(const Vector2& duv) const
{
    auto Mitchell1D = [this](Float x)
    {
        x = Float(2) * x * radiusRecip;
        x = std::abs(x);
        Float x2 = x * x;
        Float x3 = x2 * x;

        Vector4 coeffs = Vector4::Zero();
        if(x < 1)       coeffs = coeffs01;
        else if(x < 2)  coeffs = coeffs12;

        Float result = (coeffs[0] * x3 +
                        coeffs[1] * x2 +
                        coeffs[2] * x +
                        coeffs[3]);
        return result * Float(0.166666) * radius;
    };
    Vector2 mitchell2D = Vector2(Mitchell1D(duv[0]), Mitchell1D(duv[1]));
    return mitchell2D.Multiply();
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector2> MitchellNetravaliFilter::Sample(const Vector2& xi) const
{
    // And here we go
    // Couldn't find a sampling routine for M-N filter.
    // Easy to calculate derivative for CDF, but dunno how to invert that thing.
    // Instead, I'll do a MIS sampling with 3 gaussians. (We could do a uniform
    // sampling and call it a day but w/e)
    //
    // Hand crafted two gaussians, one for the middle part and other for the negative
    // this is only usable for b = 0.333, c = 0.333. It may get worse when b/c changes.
    // I don't know how long desmos links stay, but here is the calculation.
    // https://www.desmos.com/calculator/ijr8qi2dcy

    // Sampling is somewhat costly here, instead of doing MIS for each dimension,
    // sample in spherical coordinates (disk)
    //using namespace Distribution::Common;
    //auto thetaSample = SampleUniformRange(xi[0], 0, MathConstants::Pi<Float>());
    //std::array<Float, 3> weights{MIS_SIDES, MIS_MID, MIS_SIDES};
    //auto [index, localXi] = BisectSample(xi[1], Span<Float, 3>(weights.data(), 3), true);
    //std::array<Float, 3> pdfs;
    //Float radiusSample;
    //switch(index)
    //{
    //    case 0:
    //    {
    //        auto r = SampleGaussian(localXi, sideSigma, -sideMean);
    //        pdfs[0] = r.pdf;
    //        pdfs[1] = PDFGaussian(r.value, midSigma);
    //        pdfs[2] = PDFGaussian(r.value, sideSigma, sideMean);
    //        radiusSample = r.value;
    //        break;
    //    }
    //    case 1:
    //    {
    //        auto r = SampleGaussian(localXi, midSigma);
    //        pdfs[0] = PDFGaussian(r.value, sideSigma, -sideMean);
    //        pdfs[1] = r.pdf;
    //        pdfs[2] = PDFGaussian(r.value, sideSigma, sideMean);
    //        radiusSample = r.value;
    //        break;
    //    }
    //    case 2:
    //    {
    //        auto r = SampleGaussian(localXi, sideSigma, sideMean);
    //        pdfs[0] = PDFGaussian(r.value, sideSigma, -sideMean);
    //        pdfs[1] = PDFGaussian(r.value, midSigma);
    //        pdfs[2] = r.pdf;
    //        radiusSample = r.value;
    //        break;
    //    }
    //    default: assert(false);
    //}
    //using namespace Distribution;
    //Float misWeight = MIS::BalanceCancelled(Span<Float, 3>(pdfs.data(), 3),
    //                                        Span<Float, 3>(weights.data(), 3));

    //// Convert
    //using namespace Graphics;
    //Vector2 xy = PolarToCartesian(Vector2(radiusSample, thetaSample.value));
    //return SampleT<Vector2>
    //{
    //    .value = xy,
    //    .pdf = misWeight
    //};

    // Sample X
    auto SampleDim = [this](Float xi)
    {
        using namespace Distribution::Common;
        std::array<Float, 3> weights{MIS_SIDES, MIS_MID, MIS_SIDES};
        auto [index, localXi] = BisectSample(xi, Span<Float, 3>(weights.data(), 3), true);

        Float sampleVal;
        std::array<Float, 3> pdfs;
        switch(index)
        {
            case 0:
            {
                auto r = SampleGaussian(localXi, sideSigma, -sideMean);
                pdfs[0] = r.pdf;
                pdfs[1] = PDFGaussian(r.value, midSigma);
                pdfs[2] = PDFGaussian(r.value, sideSigma, sideMean);
                sampleVal = r.value;
                break;
            }
            case 1:
            {
                auto r = SampleGaussian(localXi, midSigma);
                pdfs[0] = PDFGaussian(r.value, sideSigma, -sideMean);
                pdfs[1] = r.pdf;
                pdfs[2] = PDFGaussian(r.value, sideSigma, sideMean);
                sampleVal = r.value;
                break;
            }
            case 2:
            {
                auto r = SampleGaussian(localXi, sideSigma, sideMean);
                pdfs[0] = PDFGaussian(r.value, sideSigma, -sideMean);
                pdfs[1] = PDFGaussian(r.value, midSigma);
                pdfs[2] = r.pdf;
                sampleVal = r.value;
                break;
            }
            default: assert(false);
        }
        using namespace Distribution;
        Float misWeight = MIS::BalanceCancelled(Span<Float, 3>(pdfs.data(), 3),
                                                Span<Float, 3>(weights.data(), 3));
        return SampleT<Float>
        {
            .value = sampleVal,
            .pdf = misWeight
        };
    };

    auto s0 = SampleDim(xi[0]);
    auto s1 = SampleDim(xi[1]);
    return SampleT<Vector2>
    {
        .value = Vector2(s0.value, s1.value),
        .pdf = s0.pdf * s1.pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float MitchellNetravaliFilter::Pdf(const Vector2& duv) const
{
    using namespace Distribution;
    Float x = duv.Length();
    std::array<Float, 3> weights{MIS_SIDES, MIS_MID, MIS_SIDES};
    std::array<Float, 3> pdfs
    {
        Common::PDFGaussian(x, sideSigma, -sideMean),
        Common::PDFGaussian(x, midSigma),
        Common::PDFGaussian(x, sideSigma, sideMean)
    };
    return MIS::BalanceCancelled(Span<Float, 3>(pdfs.data(), 3),
                                 Span<Float, 3>(weights.data(), 3));
}