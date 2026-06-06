#pragma once

#include "Core/Vector.h"
#include "DistributionFunctions.h"

class BoxFilter1D
{
    public:
    static constexpr Float IDEAL_RADIUS = Float(0.5);

    private:
    Float   radius;
    Float   recipRadius;

    public:
    // Constructors & Destructor
    MR_PF_DECL_V    BoxFilter1D(Float radius);
    //
    MR_PF_DECL
    Float           Evaluate(Float du) const;
    MR_PF_DECL
    SampleT<Float>  Sample(Float xi) const;
    MR_PF_DECL
    Float           Pdf(Float du) const;
    MR_PF_DECL
    Float           Radius() const;
};

class TentFilter1D
{
    public:
    static constexpr Float IDEAL_RADIUS = 1;

    private:
    Float radius;
    Float recipRadius;

    public:
    // Constructors & Destructor
    MR_PF_DECL_V    TentFilter1D(Float radius);
    //
    MR_PF_DECL
    Float           Evaluate(Float du) const;
    MR_PF_DECL
    SampleT<Float>  Sample(Float xi) const;
    MR_PF_DECL
    Float           Pdf(Float du) const;
    MR_PF_DECL
    Float           Radius() const;
};

class GaussianFilter1D
{
    public:
    static constexpr Float IDEAL_RADIUS = Float(2);

    private:
    Float radius;
    Float sigma;

    public:
    // Constructors & Destructor
    MR_PF_DECL_V    GaussianFilter1D(Float radius);
    //
    MR_PF_DECL
    Float           Evaluate(Float du) const;
    MR_PF_DECL
    SampleT<Float>  Sample(Float xi) const;
    MR_PF_DECL
    Float           Pdf(Float du) const;
    MR_PF_DECL
    Float           Radius() const;
};

class MitchellNetravaliFilter1D
{
    public:
    static constexpr Float IDEAL_RADIUS = 2;

    private:
    // Ratios are calculated like this
    // https://www.desmos.com/calculator/vkyenthaiq
    static constexpr auto MIS_MID       = Float(0.960566188838);
    static constexpr auto MIS_SIDES     = Float(0.0197169055809);

    static constexpr auto SIDE_MEAN     = Float(1.3);
    static constexpr auto SIDE_STD_DEV  = Float(0.2);
    static constexpr auto MID_STD_DEV   = Float(0.528);
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
    MR_PF_DECL_V    MitchellNetravaliFilter1D(Float radius,
                                              Float b = Float(0.33333),
                                              Float c = Float(0.33333));
    //
    MR_PF_DECL
    Float           Evaluate(Float du) const;
    MR_PF_DECL
    SampleT<Float>  Sample(Float xi) const;
    MR_PF_DECL
    Float           Pdf(Float du) const;
    MR_PF_DECL
    Float           Radius() const;
};

template<class Filter>
class SeperableFilter2D
{
    public:
    using Filter1D = Filter;
    static constexpr Float IDEAL_RADIUS = Filter1D::IDEAL_RADIUS;

    private:
    Filter1D            filter;

    public:
    // Constructors & Destructor
    MR_PF_DECL_V        SeperableFilter2D(Float radius);
    //
    MR_PF_DECL
    Float               Evaluate(const Vector2& duv) const;
    MR_PF_DECL
    SampleT<Vector2>    Sample(const Vector2& xi) const;
    MR_PF_DECL
    Float               Pdf(const Vector2& duv) const;
    MR_PF_DECL
    Float               Radius() const;
};

MR_PF_DEF_V
BoxFilter1D::BoxFilter1D(Float r)
    : radius(r)
    , recipRadius(Float(1) / radius)
{}

MR_PF_DEF
Float BoxFilter1D::Evaluate(Float du) const
{
    Float t = Math::Abs(du);
    Float rr = recipRadius;
    return (t <= radius) ? (Float(0.5) * rr)
                         : Float(0);
}

MR_PF_DEF
SampleT<Float> BoxFilter1D::Sample(Float xi) const
{
    using namespace Distribution;
    return Common::SampleUniformRange(xi, -radius, radius);
}

MR_PF_DEF
Float BoxFilter1D::Pdf(Float du) const
{
    using namespace Distribution;
    return Common::PDFUniformRange(du, -radius, radius);
}

MR_PF_DEF
Float BoxFilter1D::Radius() const
{
    return radius;
}

MR_PF_DEF_V
TentFilter1D::TentFilter1D(Float r)
    : radius(r)
    , recipRadius(Float(1) / radius)
{}

MR_PF_DEF
Float TentFilter1D::Evaluate(Float du) const
{
    using namespace Math;
    Float t = Abs(du * recipRadius);
    Float cap = recipRadius;
    Float x = Lerp<Float>(cap, 0, t);
    return x;
}

MR_PF_DEF
SampleT<Float> TentFilter1D::Sample(Float xi) const
{
    using namespace Distribution;
    return Common::SampleTent(xi, -radius, radius);
}

MR_PF_DEF
Float TentFilter1D::Pdf(Float du) const
{
    using namespace Distribution;
    return Common::PDFTent(du, -radius, radius);
}

MR_PF_DEF
Float TentFilter1D::Radius() const
{
    return radius;
}

MR_PF_DEF_V
GaussianFilter1D::GaussianFilter1D(Float r)
    : radius(r)
    // ~%99.5 of samples lies between [-r,r]
    , sigma(r * Float(0.285714))
{}

MR_PF_DEF
Float GaussianFilter1D::Evaluate(Float du) const
{
    return Math::Gaussian(du, sigma);
}

MR_PF_DEF
SampleT<Float> GaussianFilter1D::Sample(Float xi) const
{
    using namespace Distribution;
    return Common::SampleGaussian(xi, sigma);
}

MR_PF_DEF
Float GaussianFilter1D::Pdf(Float du) const
{
    using namespace Distribution;
    return Common::PDFGaussian(du, sigma);
}

MR_PF_DEF
Float GaussianFilter1D::Radius() const
{
    return radius;
}

MR_PF_DEF_V
MitchellNetravaliFilter1D::MitchellNetravaliFilter1D(Float r, Float b, Float c)
    : radius(r)
    , radiusRecip(Float(1) / radius)
    , midSigma(MID_STD_DEV * r * Float(0.5))
    , sideSigma(SIDE_STD_DEV * r * Float(0.5))
    , sideMean(SIDE_MEAN * r * Float(0.5))
{
    constexpr Float F = Float(1) / Float(6);
    // This gives the exact integral to be 1
    coeffs01[0] = F * (Float(12) - Float(9) * b - Float(6) * c);
    coeffs01[1] = F * (Float(-18) + Float(12) * b + Float(6) * c);
    coeffs01[2] = Float(0);
    coeffs01[3] = F * (Float(6) - Float(2) * b);

    coeffs12[0] = F * (-b - Float(6) * c);
    coeffs12[1] = F * (Float(6) * b + Float(30) * c);
    coeffs12[2] = F * (Float(-12) * b - Float(48) * c);
    coeffs12[3] = F * (Float(8) * b + Float(24) * c);
}

MR_PF_DEF
Float MitchellNetravaliFilter1D::Evaluate(const Float du) const
{
    Float x = du;
    x = Float(2) * x * radiusRecip;
    x = Math::Abs(x);

    Vector4 coeffs = Vector4::Zero();
         if(x < 1)  coeffs = coeffs01;
    else if(x < 2)  coeffs = coeffs12;

    Float result;
    result = Math::FMA(coeffs[0], x, coeffs[1]);
    result = Math::FMA(result   , x, coeffs[2]);
    result = Math::FMA(result   , x, coeffs[3]);
    return result * Float(2) * radiusRecip;
}

MR_PF_DEF
SampleT<Float> MitchellNetravaliFilter1D::Sample(Float xi) const
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
    // https://www.desmos.com/calculator/vkyenthaiq
    //
    using namespace Distribution::Common;
    Array<Float, 3> weights{MIS_SIDES, MIS_MID, MIS_SIDES};
    auto [index, localXi] = BisectSample<3>(xi, Span<Float, 3>(weights.data(), 3), true);

    Float sampleVal = Float(0);
    Array<Float, 3> pdfs;
    assert(index <= 2);
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
        default:
        {
            pdfs = {};
            assert(false);
        }
    }
    using namespace Distribution;
    Float misWeight = MIS::BalanceCancelled<3>(pdfs, weights);
    return SampleT<Float>
    {
        .value = sampleVal,
        .pdf = misWeight
    };
}

MR_PF_DEF
Float MitchellNetravaliFilter1D::Pdf(Float du) const
{
    using namespace Distribution;
    Array<Float, 3> weights{MIS_SIDES, MIS_MID, MIS_SIDES};
    Array<Float, 3> pdfs =
    {
        Common::PDFGaussian(du, sideSigma, -sideMean),
        Common::PDFGaussian(du, midSigma),
        Common::PDFGaussian(du, sideSigma, sideMean)
    };
    Float mis = MIS::BalanceCancelled<3>(pdfs, weights);
    return mis;
}

MR_PF_DEF
Float MitchellNetravaliFilter1D::Radius() const
{
    return radius;
}

template<class F>
MR_PF_DEF_V
SeperableFilter2D<F>::SeperableFilter2D(Float radius)
    : filter(radius)
{}

template<class F>
MR_PF_DEF
Float SeperableFilter2D<F>::Evaluate(const Vector2& duv) const
{
    return filter.Evaluate(duv[0]) * filter.Evaluate(duv[1]);
}

template<class F>
MR_PF_DEF
SampleT<Vector2> SeperableFilter2D<F>::Sample(const Vector2& xi) const
{
    auto s0 = filter.Sample(xi[0]);
    auto s1 = filter.Sample(xi[1]);
    return SampleT<Vector2>
    {
        .value = Vector2(s0.value, s1.value),
        .pdf   = s0.pdf * s1.pdf
    };
}

template<class F>
MR_PF_DEF
Float SeperableFilter2D<F>::Pdf(const Vector2& duv) const
{
    return filter.Pdf(duv[0]) * filter.Pdf(duv[1]);
}

template<class F>
MR_PF_DEF
Float SeperableFilter2D<F>::Radius() const
{
    return filter.Radius();
}

using BoxFilter               = SeperableFilter2D<BoxFilter1D>;
using TentFilter              = SeperableFilter2D<TentFilter1D>;
using GaussianFilter          = SeperableFilter2D<GaussianFilter1D>;
using MitchellNetravaliFilter = SeperableFilter2D<MitchellNetravaliFilter1D>;
