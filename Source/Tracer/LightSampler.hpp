#pragma once

#include "LightSampler.h"

MR_HF_DEF
Pair<Ray, Vector2> LightSampleOutput::SampledRay(const Vector3& distantPoint) const
{
    using RetT = Pair<Ray, Vector2>;
    Vector3 dir = Math::Normalize(position - distantPoint);
    // Nudge the position back here, this will be used
    // for visibility check
    // TODO: Bad usage of API, constructing a ray to nudge a position
    Vector3 pos = Ray(Vector3::Zero(), position).Nudge(-dir).Pos();
    Float length = Math::Length(pos - distantPoint);
    return RetT(Ray(dir, distantPoint), Vector2(0, length * 0.999f));
}

template<class ML>
inline DirectLightSamplerUniform<ML>::DirectLightSamplerUniform(const MetaLightArrayView& la,
                                                                const LightLookupTable& lt)
    : dMetaLights(la)
    , dLightIndexTable(lt)
{}

template <class ML>
template <class SpectrumConverter>
MR_HF_DEF
LightSample DirectLightSamplerUniform<ML>::SampleLight(RNGDispenser& rng,
                                                       const SpectrumConverter& stConverter,
                                                       const Vector3& distantPoint,
                                                       const Vector3& distantNormal,
                                                       const RayConeSurface& distantRayConeSurf) const
{
    using MetaLightView = typename MetaLightArrayView::template MetaLightView<SpectrumConverter>;
    static constexpr auto DiscreteSampleIndex = ML::SampleSolidAngleRNListWorst.TotalRNCount();

    uint32_t lightCount = dMetaLights.Size();
    Float lightCountF = Float(lightCount);
    // Randomly Select Light
    Float dXi = rng.NextFloat<DiscreteSampleIndex>();
    Float lightIndexF = dXi * lightCountF;
    uint32_t lightIndex = static_cast<uint32_t>(lightIndexF);
    // Rarely, index becomes the light count
    // due to precision.
    lightIndex = Math::Clamp(lightIndex, 0u, lightCount - 1u);
    // Actual sampling
    MetaLightView metaLight = dMetaLights(stConverter, lightIndex);
    SampleT<Vector3> pointSample = metaLight.SampleSolidAngle(rng, distantPoint);

    Vector3 wO = (distantPoint - pointSample.value);
    Float distance = Math::Length(wO);
    wO = Math::Normalize(wO);
    RayCone surfaceCone = distantRayConeSurf.ConeAfterScatter(wO, distantNormal);
    surfaceCone = surfaceCone.Advance(distance);
    Spectrum emission = metaLight.EmitViaSurfacePoint(wO, pointSample.value,
                                                      surfaceCone);

    Float selectionPdf = Float(1) / lightCountF;
    return LightSample
    {
        .value = LightSampleOutput
        {
            .lightIndex = lightIndex,
            .position = pointSample.value,
            .emission = emission
        },
        .pdf = selectionPdf * pointSample.pdf
    };
}

template <class ML>
MR_HF_DEF
Float DirectLightSamplerUniform<ML>::PdfLight(uint32_t index,
                                              const MetaHit& hit,
                                              const Ray& ray) const
{
    assert(index < dMetaLights.Size());
    using SCIdentity = SpectrumContextIdentity;
    using MetaLightView = typename MetaLightArrayView::template MetaLightView<SCIdentity >;
    SCIdentity specContext;

    // Discrete sampling of such light (its uniform)
    uint32_t lightCount = dMetaLights.Size();
    Float lightCountF = Float(lightCount);
    Float selectionPDF = Float(1) / lightCountF;

    // Probability of sampling such direction from the particular light
    MetaLightView light = dMetaLights(specContext, index);
    Float lightPDF = light.PdfSolidAngle(hit, ray.Pos(), ray.Dir());
    return lightPDF * selectionPDF;
}

template <class ML>
MR_HF_DEF
Float DirectLightSamplerUniform<ML>::PdfLight(const HitKeyPack& hitPack,
                                              const MetaHit& hit,
                                              const Ray& r) const
{
    const auto& lmK = hitPack.lightOrMatKey;
    LightKey lK = LightKey::CombinedKey(lmK.FetchBatchPortion(),
                                        lmK.FetchIndexPortion());

    LightSurfKeyPack keyPack =
    {
        .lK = std::bit_cast<CommonKey>(lK),
        .tK = std::bit_cast<CommonKey>(hitPack.transKey),
        .pK = std::bit_cast<CommonKey>(hitPack.primKey)
    };
    auto lightIndexOpt = dLightIndexTable.Search(keyPack);
    if(lightIndexOpt.has_value())
        return PdfLight(lightIndexOpt.value(), hit, r);

    return Float(0);
}
