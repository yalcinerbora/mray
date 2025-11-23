#pragma once

#include "MediumsDefault.h"
#include "DistributionFunctions.h"

namespace MediumDetail
{

MR_PF_DEF
bool SingleSegmentIterator::Advance()
{
    return false;
}

template<class S>
MR_HF_DEF
MediumTraverser<S>::MediumTraverser(const Ray&, const Vector2&,
                                    const SegmentIterator& it)
    : it(it)
    , dt(0)
{}

template<class S>
MR_HF_DEF
bool MediumTraverser<S>::SampleTMajor(Spectrum& tMaj, Spectrum& sMaj,
                                      Float& t, Float xi)
{
    using Distribution::Common::SampleExp;
    const auto& segment = it.curSegment;
    bool isTerminated = false;

    if(segment.tMM[0] + dt > segment.tMM[1])
        isTerminated = it.Advance();

    // Set iteration state
    dt = SampleExp(xi, segment.sMajor[0]).value;
    sMaj = segment.sMajor;
    t = segment.tMM[0] + dt;
    tMaj = Math::Exp(-dt * segment.sMajor);

    return !isTerminated;
}

//===========================//
//           Vacuum          //
//===========================//
template <class SC>
MR_PF_DEF_V
MediumVacuum<SC>::MediumVacuum(const SpectrumConverter&,
                               const DataSoA&, MediumKey) noexcept
{}

template <class SC>
MR_PF_DEF
ScatterSample MediumVacuum<SC>::SampleScattering(const Vector3&, const Vector3&,
                                                 RNGDispenser&) const noexcept
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

template <class SC>
MR_PF_DEF
Float MediumVacuum<SC>::PdfScattering(const Vector3&, const Vector3&,
                                      const Vector3&) const noexcept
{
    return Float(0.0);
}

template <class SC>
MR_PF_DEF
Spectrum MediumVacuum<SC>::SigmaA(const Vector3&) const noexcept
{
    return Spectrum::Zero();
}

template <class SC>
MR_PF_DEF
Spectrum MediumVacuum<SC>::SigmaS(const Vector3&) const noexcept
{
    return Spectrum::Zero();
}

template <class SC>
MR_PF_DEF
Spectrum MediumVacuum<SC>::Emission(const Vector3&) const noexcept
{
    return Spectrum::Zero();
}

template <class SC>
MR_PF_DEF
bool MediumVacuum<SC>::HasEmission() const
{
    return false;
}

template <class SC>
MR_HF_DEF
typename MediumVacuum<SC>::Traverser
MediumVacuum<SC>::GenTraverser(const Ray& r, const Vector2& tMM) const
{
    return MediumTraverser(r, tMM, SingleSegmentIterator
    {
        .curSegment =
        {
            .tMM = tMM,
            .sMajor = Spectrum(0)
        }
    });
}

//===========================//
//        Homogeneous        //
//===========================//
template <class SC>
MR_HF_DEF
MediumHomogeneous<SC>::MediumHomogeneous(const SpectrumConverter& sc,
                                         const DataSoA& soa, MediumKey k)
    : sigmaA(sc.ConvertAlbedo(soa.Get<SIGMA_A>()[k.FetchIndexPortion()]))
    , sigmaS(sc.ConvertAlbedo(soa.Get<SIGMA_S>()[k.FetchIndexPortion()]))
    , emission(sc.ConvertRadiance(soa.Get<EMISSION>()[k.FetchIndexPortion()]))
    , g(soa.Get<HG_PHASE>()[k.FetchIndexPortion()])
{}

template <class SC>
MR_HF_DEF
ScatterSample MediumHomogeneous<SC>::SampleScattering(const Vector3& wO,
                                                      const Vector3&,
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

template <class SC>
MR_HF_DEF
Float MediumHomogeneous<SC>::PdfScattering(const Vector3& wI,
                                           const Vector3& wO,
                                           const Vector3&) const
{
    using namespace Distribution::Medium;
    Float cosTheta = Math::Dot(wI, wO);
    return HenyeyGreensteinPhase(cosTheta, g);
}

template <class SC>
MR_HF_DEF
Spectrum MediumHomogeneous<SC>::SigmaA(const Vector3&) const
{
    return sigmaA;
}

template <class SC>
MR_HF_DEF
Spectrum MediumHomogeneous<SC>::SigmaS(const Vector3&) const
{
    return sigmaS;
}

template <class SC>
MR_HF_DEF
Spectrum MediumHomogeneous<SC>::Emission(const Vector3&) const
{
    return emission;
}

template <class SC>
MR_PF_DEF
bool MediumHomogeneous<SC>::HasEmission() const
{
    return emission != Spectrum::Zero();
}

template <class SC>
MR_HF_DEF
typename MediumHomogeneous<SC>::Traverser
MediumHomogeneous<SC>::GenTraverser(const Ray& r, const Vector2& tMM) const
{
    return MediumTraverser(r, tMM, SingleSegmentIterator
    {
        .curSegment =
        {
            .tMM = tMM,
            .sMajor = sigmaA + sigmaS
        }
    });
}

}