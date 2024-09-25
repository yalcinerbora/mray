#pragma once

namespace MediumDetail
{

//===========================//
//           Vacuum          //
//===========================//
MRAY_HYBRID MRAY_CGPU_INLINE
MediumVacuum::MediumVacuum(const SpectrumConverter&,
                           const DataSoA&, MediumKey)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
ScatterSample MediumVacuum::SampleScattering(const Vector3&, RNGDispenser&) const
{
    return ScatterSample
    {
        .value =
        {
            .wI = Vector3::Zero(),
            .phaseVal = Float(0.0)
        },
        .pdf = Float(0.0)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float MediumVacuum::PdfScattering(const Vector3&, const Vector3&) const
{
    return Float(0.0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::IoR(const Vector3&) const
{
    return Spectrum(1.0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::SigmaA(const Vector3&) const
{
    return Spectrum::Zero();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::SigmaS(const Vector3&) const
{
    return Spectrum::Zero();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::Emission(const Vector3&) const
{
    return Spectrum::Zero();
}

//===========================//
//        Homogeneous        //
//===========================//
template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
MediumHomogeneous<ST>::MediumHomogeneous(const SpectrumConverter& sc,
                                         const DataSoA& soa, MediumKey k)
    : sigmaA(sc.Convert(soa.Get<SIGMA_A>()[k.FetchIndexPortion()]))
    , sigmaS(sc.Convert(soa.Get<SIGMA_S>()[k.FetchIndexPortion()]))
    , emission(sc.Convert(soa.Get<EMISSION>()[k.FetchIndexPortion()]))
    , g(soa.Get<HG_PHASE>()[k.FetchIndexPortion()])
{
    // TODO: Check if this is state of the art?
    // Check manuals of renderers and papers
    //
    // First parameter of IoR input is middle IoR (550nm)
    // Last two parameters are Cauchy coefficients (only first two A and B)
    Vector3 iorIn = soa.Get<IOR>()[k.FetchIndexPortion()];
    using namespace Graphics;
    if constexpr(std::is_same_v<ST, SpectrumConverterContextIdentity>)
        ior = Spectrum(iorIn[0]);
    else
        ior = WavesToSpectrumCauchy(sc.Waves(),
                                    Vector2(iorIn[1], iorIn[2]));
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
ScatterSample MediumHomogeneous<ST>::SampleScattering(const Vector3& wO,
                                                      RNGDispenser& rng) const
{
    using namespace MediumFunctions;
    Vector2 xi = rng.NextFloat2D<0>();
    // We send the wO is world space
    // So returned vector is in world space
    // TODO: Check this
    auto hgSample = SampleHenyeyGreensteinPhase(wO, g, xi);
    return ScatterSample
    {
        .value =
        {
            .wI = hgSample.value,
            .phaseVal = hgSample.pdf
        },
        .pdf = hgSample.pdf
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MediumHomogeneous<ST>::PdfScattering(const Vector3& wI,
                                           const Vector3& wO) const
{
    using namespace MediumFunctions;
    Float cosTheta = wI.Dot(wO);
    return HenyeyGreensteinPhase(cosTheta, g);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumHomogeneous<ST>::IoR(const Vector3&) const
{
    return ior;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumHomogeneous<ST>::SigmaA(const Vector3&) const
{
    return sigmaA;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumHomogeneous<ST>::SigmaS(const Vector3&) const
{
    return sigmaS;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumHomogeneous<ST>::Emission(const Vector3&) const
{
    return emission;
}

}