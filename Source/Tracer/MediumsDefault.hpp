#pragma once

#include "MediumsDefault.h"
#include "DistributionFunctions.h"
#include "StochasticTexFilter.h"

namespace MediumDetail
{

template<uint32_t B>
MR_PF_DEF
uint32_t DenseDDAIterator<B>::SelectAxis() const
{
    // PBRT does a look-up table.
    // Ggot bored, and wanted to do K-Map optimization.
    // Maybe faster? At least it has no LUT so
    // probably less register pressure (given compiler could not optimize).
    //
    // Invert the comparisons according to direction sign
    Float x = Math::SignPM1(deltaT[0]) * nextAxes[0];
    Float y = Math::SignPM1(deltaT[1]) * nextAxes[1];
    Float z = Math::SignPM1(deltaT[2]) * nextAxes[2];

    // Inputs
    // Comparison table is from PBRT (I mean, you can generate it as well...)
    // https://www.pbr-book.org/4ed/Volume_Scattering/Media#DDAMajorantIterator
    //
    //  x < y | x < z | y < z || ???
    // ------------------------------
    //    0   |   0   |   0   ||  Z
    //    0   |   0   |   1   ||  Y
    //    0   |   1   |   0   ||  -  --> This does not makes sense (was Z).
    //    0   |   1   |   1   ||  Y
    //    1   |   0   |   0   ||  Z
    //    1   |   0   |   1   ||  -  --> This does not makes sense (was Z).
    //    1   |   1   |   0   ||  X
    //    1   |   1   |   1   ||  X
    bool a = x < y;
    bool b = x < z;
    bool c = y < z;
    // Outputs
    // X_0 (LSB of the Index)
    //
    // K-Map:
    // A \ BC | 00 | 01 | 11 | 10
    // ---------------------------
    //   0    | 0  | 1  | 1  | X
    // ---------------------------
    //   1    | 0  | X  | 0  | 0
    //
    // Which results in to:
    // A'C
    bool x0 = !a && c;

    // X_1 (MSB of the Index)
    //
    // K-Map:
    // A \ BC | 00 | 01 | 11 | 10
    // ---------------------------
    //   0    | 1  | 0  | 0  | X
    // ---------------------------
    //   1    | 1  | X  | 0  | 0
    //
    // Which results in to:
    // A'C' + AB'
    bool x1 = (!a && !a) || (a && !b);

    return uint32_t(x0) + (uint32_t(x1) << 1);
}

template<uint32_t B>
MR_GF_DEF
DenseDDAIterator<B>::DenseDDAIterator(const TracerTexView<3, Float>& t,
                                      Spectrum sigmaT,
                                      const Ray& rayIn,
                                      const Vector2& tMMIn)
    : majDensityTex(t)
    , r(rayIn)
    , sigmaT(sigmaT)
    , deltaT(Vector3(DELTA_XYZ) / rayIn.dir)
    , tMax(tMMIn[1])
{
    // TODO: Clamp the tMM for termination
    // !!!!
    //
    // From PBRT, but it is register optmized a little
    // we on-the-fly do the step parameter.
    // Also we utilize the texture system to auto nearest sample
    // by the uv coordinates.
    //
    // Rays always start inside / near grid.
    curSegment.tMM[0] = tMMIn[0];
    Vector3 curP = rayIn.AdvancedPos(tMMIn[0]);
    nextAxes = curP + deltaT;
    //
    Float m = Math::Min(Math::Abs(deltaT[0]), Math::Abs(deltaT[1]));
    curSegment.tMM[1] = Math::Min(m, Math::Abs(deltaT[2]));
}

template<uint32_t B>
MR_GF_DEF
bool DenseDDAIterator<B>::Advance()
{
    uint32_t aI = SelectAxis();
    Float step = Math::Min(tMax, deltaT[aI]);
    nextAxes[aI] += deltaT[aI];

    if(curSegment.tMM[1] >= tMax)
        return false;
    // We are averaging t here to make calculations on the edge
    // little bit more numerically confined maybe?
    Vector3 uv = r.AdvancedPos(curSegment.tMM.Sum() * Float(0.5));
    curSegment.sMajor = majDensityTex(uv) * sigmaT;
    curSegment.tMM[0] = curSegment.tMM[1];
    curSegment.tMM[1] += step;
    //
    return true;
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
    , emission(Spectrum::Zero())
    , g(soa.Get<HG_PHASE>()[k.FetchIndexPortion()])
{
    Vector3 emissionRGB = soa.Get<EMISSION>()[k.FetchIndexPortion()];
    if(emissionRGB != Vector3::Zero())
        emission = sc.ConvertRadiance(emissionRGB);
}

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
MR_PF_DECL
Float MediumHomogeneous<SC>::EvalScattering(const Vector3& wI,
                                            const Vector3& wO,
                                            const Vector3& p) const noexcept
{
    return PdfScattering(wI, wO, p);
}

template <class SC>
MR_HF_DECL
MediumQuery MediumHomogeneous<SC>::Query(const Vector3&, Float) const
{
    return MediumQuery
    {
        .sigmaA   = sigmaA,
        .sigmaS   = sigmaS,
        .emission = emission
    };
}

template <class SC>
MR_HF_DEF
typename MediumHomogeneous<SC>::Traverser
MediumHomogeneous<SC>::GenTraverser(const Ray&, const Vector2& tMM) const
{
    return MediumTraverser(SingleSegmentIterator
    {
        .curSegment =
        {
            .tMM = tMM,
            .sMajor = sigmaA + sigmaS
        }
    });
}

//===========================//
//       Heterogeneous       //
//===========================//
template <class SC>
MR_HF_DEF
MediumHeterogeneous<SC>::MediumHeterogeneous(const SpectrumConverter& sc,
                                             const DataSoA& soa, MediumKey k)
    : sigmaA(sc.ConvertAlbedo(soa.Get<SIGMA_A>()[k.FetchIndexPortion()]))
    , sigmaS(sc.ConvertAlbedo(soa.Get<SIGMA_S>()[k.FetchIndexPortion()]))
    , phaseG(soa.Get<HG_PHASE>()[k.FetchIndexPortion()])
    , tempatureRange(soa.Get<TEMPATURE_RANGE>()[k.FetchIndexPortion()])
    , densityMap(soa.Get<DENSITY>()[k.FetchIndexPortion()])
    , tempatureMap(soa.Get<TEMPATURE>()[k.FetchIndexPortion()])
    , majMap(soa.Get<MAJORANT>()[k.FetchIndexPortion()])
    , topology(soa.Get<TOPOLOGY>()[k.FetchIndexPortion()])
    , sc(&sc)
{}

template <class SC>
MR_HF_DEF
ScatterSample MediumHeterogeneous<SC>::SampleScattering(const Vector3& wO,
                                                        const Vector3&,
                                                        RNGDispenser& rng) const
{
    using namespace Distribution::Medium;
    Vector2 xi = rng.NextFloat2D<0>();
    auto hgSample = SampleHenyeyGreensteinPhase(wO, phaseG, xi);
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
Float MediumHeterogeneous<SC>::PdfScattering(const Vector3& wI,
                                             const Vector3& wO,
                                             const Vector3&) const
{
    using namespace Distribution::Medium;
    Float cosTheta = Math::Dot(wI, wO);
    return HenyeyGreensteinPhase(cosTheta, phaseG);
}

template <class SC>
MR_PF_DECL
Float MediumHeterogeneous<SC>::EvalScattering(const Vector3& wI,
                                              const Vector3& wO,
                                              const Vector3& p) const noexcept
{
    return PdfScattering(wI, wO, p);
}

template <class SC>
MR_GF_DECL
MediumQuery MediumHeterogeneous<SC>::Query(const Vector3& p, Float xi) const
{
    static constexpr auto EMPTY = VolumetricSVO::VolGrid6_2::EMPTY_INDEX_VAL;
    // Stochastic tri-cubic filter
    Vector3 qPoint = StochasticTF::Tricubic(p * topology.Resolution(), xi);
    //
    uint32_t dataIndex = topology(Vector3ui(qPoint));
    bool isEmpty = (dataIndex != EMPTY);
    //
    Float density = (isEmpty) ? densityMap(dataIndex) : Float(0);
    MediumQuery result = MediumQuery
    {
        .sigmaA = sigmaA * density,
        .sigmaS = sigmaS * density,
        .emission = std::nullopt
    };

    if(tempatureMap && !isEmpty)
    {
        // This is [0, 1]
        Float temp01 = (dataIndex != EMPTY) ? tempatureMap.Value()(dataIndex) : Float(0);
        Float size = tempatureRange[1] - tempatureRange[0];
        // This Plank's Law-feedable value (in kelvins)
        Float tempature = temp01 * size + tempatureRange[0];
        Spectrum emission = Spectrum::Zero();

        //
        static constexpr auto WAVE_COUNT = SpectrumConverter::IsRGB
                                            ? uint32_t(3)
                                            : SpectraPerSpectrum;
        SpectrumWaves waves = sc->Wavelengths();
        for(uint32_t i = 0; i < WAVE_COUNT; i++)
            emission[i] = BlackbodySPD::PlancksLaw(waves.GetWl(i), tempature);

        result.emission = emission;
    }
    return result;
}

template <class SC>
MR_HF_DEF
typename MediumHeterogeneous<SC>::Traverser
MediumHeterogeneous<SC>::GenTraverser(const Ray& r, const Vector2& tMM) const
{
    using DDA = DenseDDAIterator<GRID_BITS>;
    return MediumHeterogeneous<SC>::Traverser(DDA(majMap,
                                                  sigmaA + sigmaS,
                                                  r, tMM));
}

}