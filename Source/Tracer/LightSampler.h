#pragma once

#include <cstdint>
#include "Core/Vector.h"
#include "Core/DataStructures.h"

#include "Random.h"
#include "ParamVaryingData.h"

struct LightSampleOutput
{
    uint32_t    lightIndex;
    Vector3     position;
    Spectrum    emission;

    MRAY_HYBRID
    Pair<Ray, Vector2>  SampledRay(const Vector3& distantPoint) const;
};
using LightSample = SampleT<LightSampleOutput>;


template<class DLSampler>
concept DirectLightSamplerC = requires(const DLSampler& sampler,
                                       RNGDispenser& rng,
                                       const HitKeyPack& hitPack,
                                       MetaHit metaHit)
{
    {sampler.SampleLight(rng, Vector3{})} -> std::same_as<LightSample>;
    {sampler.PdfLight(uint32_t{}, metaHit, Ray{})} -> std::same_as<Float>;
    {sampler.PdfLight(hitPack, metaHit, Ray{})} -> std::same_as<Float>;
};


struct LightSurfKeyPack
{
    CommonKey lK;
    CommonKey tK;
    CommonKey pK;

    auto operator<=>(const LightSurfKeyPack&) const = default;
};

struct LightSurfKeyHasher
{
    MRAY_HYBRID
    static uint32_t Hash(const LightSurfKeyPack&);
    MRAY_HYBRID
    static bool     IsSentinel(uint32_t);
    MRAY_HYBRID
    static bool     IsEmpty(uint32_t);
};

using LightLookupTable = LookupTable<LightSurfKeyPack, uint32_t, uint32_t, 4,
                                     LightSurfKeyHasher>;

template <class MetaLightArray>
class DirectLightSamplerViewUniform
{
    private:
    MetaLightArray      dMetaLights;
    LightLookupTable    dLightIndexTable;

    public:
    template <class SpectrumConverter>
    MRAY_HYBRID
    LightSample SampleLight(RNGDispenser& rng,
                            const SpectrumConverter&,
                            const Vector3& distantPoint) const;

    MRAY_HYBRID
    Float       PdfLight(uint32_t index, const MetaHit& hit,
                         const Ray& r) const;
    MRAY_HYBRID
    Float       PdfLight(const HitKeyPack&, const MetaHit& hit,
                         const Ray& r) const;
};

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightSurfKeyHasher::Hash(const LightSurfKeyPack&)
{
    return 0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
bool LightSurfKeyHasher::IsSentinel(uint32_t hash)
{
    return hash == 0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
bool LightSurfKeyHasher::IsEmpty(uint32_t hash)
{
    return hash == 1;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<Ray, Vector2> LightSampleOutput::SampledRay(const Vector3& distantPoint) const
{
    Vector3 dir = position - distantPoint;
    // Nudge the position back here, this will be used
    // for visibility check
    // TODO: Bad usage of API, constructing a ray to nudge a position
    Vector3 pos = Ray(Vector3::Zero(), position).Nudge(-dir).Pos();
    dir = pos - distantPoint;

    Float length = dir.Length();
    dir *= (Float(1) / length);
    return Pair(Ray(dir, distantPoint), Vector2(0, length));
}

template <class ML>
template <class SpectrumConverter>
MRAY_HYBRID MRAY_CGPU_INLINE
LightSample DirectLightSamplerViewUniform<ML>::SampleLight(RNGDispenser& rng,
                                                           const SpectrumConverter& stConverter,
                                                           const Vector3& distantPoint) const
{
    using MetaLightView = typename ML::template MetaLightView<SpectrumConverter>;

    uint32_t lightCount = dMetaLights.Size();
    Float lightCountF = Float(lightCount);
    // Randomly Select Light
    Float dXi = rng.NextFloat<0>();
    Float lightIndexF = dXi * lightCountF;
    uint32_t lightIndex = static_cast<uint32_t>(lightIndexF);
    // Rarely, index becomes the light count
    // due to precision.
    lightIndex = Math::Clamp(lightIndex, 0u, lightCount - 1u);
    // Actual sampling
    MetaLightView metaLight = dMetaLights(stConverter, lightIndex);
    SampleT<Vector3> pointSample = metaLight.SampleSolidAngle(rng, distantPoint);

    Float selectionPdf = Float(1) / lightCountF;
    return LightSample
    {
        .value = LightSampleOutput
        {
            .lightIndex = lightIndex ,
            .position = pointSample.value
        },
        .pdf = selectionPdf * pointSample.pdf
    };
}

template <class ML>
MRAY_HYBRID MRAY_CGPU_INLINE
Float DirectLightSamplerViewUniform<ML>::PdfLight(uint32_t index,
                                                  const MetaHit& hit,
                                                  const Ray& ray) const
{
    using STIdentity = SpectrumConverterContextIdentity;
    using MetaLightView = typename ML::template MetaLightView<STIdentity>;
    STIdentity stIdentity;
    if(index >= dMetaLights.Size()) return Float(0);

    // Discrete sampling of such light (its uniform)
    uint32_t lightCount = dMetaLights.Size();
    Float lightCountF = Float(lightCount);
    Float selectionPDF = Float(1) / lightCountF;

    // Probability of sampling such direction from the particular light
    MetaLightView light = dMetaLights(stIdentity, index);
    Float lightPDF = light.PdfSolidAngle(hit, ray.Dir(), ray.Pos());
    return lightPDF * selectionPDF;
}

template <class ML>
MRAY_HYBRID MRAY_CGPU_INLINE
Float DirectLightSamplerViewUniform<ML>::PdfLight(const HitKeyPack& hitPack,
                                                  const MetaHit& hit,
                                                  const Ray& r) const
{
    LightSurfKeyPack keyPack =
    {
        .lK = std::bit_cast<CommonKey>(hitPack.lightOrMatKey),
        .tK = std::bit_cast<CommonKey>(hitPack.transKey),
        .pK = std::bit_cast<CommonKey>(hitPack.primKey)
    };
    auto lightIndexOpt = dLightIndexTable.Search(keyPack);
    if(lightIndexOpt.has_value())
        return PdfLight(*(lightIndexOpt.value()), hit, r);

    return Float(0);
}

#include "LightSampler.hpp"