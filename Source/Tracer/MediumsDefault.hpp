#pragma once

#include "MediumsDefault.h"
#include "DistributionFunctions.h"

namespace MediumDetail
{

//===========================//
//           Vacuum          //
//===========================//
MR_PF_DEF_V
MediumVacuum::MediumVacuum(const SpectrumConverter&,
                           const DataSoA&, MediumKey) noexcept
{}

MR_PF_DEF
ScatterSample MediumVacuum::SampleScattering(const Vector3&, RNGDispenser&) const noexcept
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

MR_PF_DEF
Float MediumVacuum::PdfScattering(const Vector3&, const Vector3&) const noexcept
{
    return Float(0.0);
}

MR_PF_DEF
Spectrum MediumVacuum::SigmaA(const Vector3&) const noexcept
{
    return Spectrum::Zero();
}

MR_PF_DEF
Spectrum MediumVacuum::SigmaS(const Vector3&) const noexcept
{
    return Spectrum::Zero();
}

MR_PF_DEF
Spectrum MediumVacuum::Emission(const Vector3&) const noexcept
{
    return Spectrum::Zero();
}

//===========================//
//        Homogeneous        //
//===========================//
template <class ST>
MR_HF_DEF
MediumHomogeneous<ST>::MediumHomogeneous(const SpectrumConverter& sc,
                                         const DataSoA& soa, MediumKey k)
    : sigmaA(sc.Convert(soa.Get<SIGMA_A>()[k.FetchIndexPortion()]))
    , sigmaS(sc.Convert(soa.Get<SIGMA_S>()[k.FetchIndexPortion()]))
    , emission(sc.Convert(soa.Get<EMISSION>()[k.FetchIndexPortion()]))
    , g(soa.Get<HG_PHASE>()[k.FetchIndexPortion()])
{}

template <class ST>
MR_HF_DEF
ScatterSample MediumHomogeneous<ST>::SampleScattering(const Vector3& wO,
                                                      RNGDispenser& rng) const
{
    using namespace Distribution::Medium;
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
MR_HF_DEF
Float MediumHomogeneous<ST>::PdfScattering(const Vector3& wI,
                                           const Vector3& wO) const
{
    using namespace Distribution::Medium;
    Float cosTheta = Math::Dot(wI, wO);
    return HenyeyGreensteinPhase(cosTheta, g);
}

template <class ST>
MR_HF_DEF
Spectrum MediumHomogeneous<ST>::SigmaA(const Vector3&) const
{
    return sigmaA;
}

template <class ST>
MR_HF_DEF
Spectrum MediumHomogeneous<ST>::SigmaS(const Vector3&) const
{
    return sigmaS;
}

template <class ST>
MR_HF_DEF
Spectrum MediumHomogeneous<ST>::Emission(const Vector3&) const
{
    return emission;
}

}