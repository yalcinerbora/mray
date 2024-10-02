#pragma once

#include <cstdint>
#include "Core/Vector.h"

#include "Random.h"
#include "ParamVaryingData.h"

struct LightSampleOutput
{
    uint32_t    lightIndex;
    Vector3     position;
};
using LightSample = SampleT<LightSampleOutput>;


template<class DLSampler>
concept DirectLightSamplerC = requires(const DLSampler& sampler,
                                       RNGDispenser& rng)
{
    {sampler.SampleLight(rng, Vector3{})} -> std::same_as<LightSample>;
    {sampler.PdfLight(uint32_t{}, Ray{})} -> std::same_as<Float>;
};

template <class MetaLightArray>
class DirectLightSamplerViewUniform
{
    private:
    MetaLightArray dMetaLights;

    public:
    MRAY_HYBRID
    LightSample SampleLight(RNGDispenser& rng,
                            const Vector3& lookPosition) const;

    MRAY_HYBRID
    Float       PdfLight(uint32_t index, const MetaHit& hit,
                         const Ray& r) const;

};

template <class ML>
MRAY_HYBRID
LightSample DirectLightSamplerViewUniform<ML>::SampleLight(RNGDispenser& rng,
                                                           const Vector3& distantPoint) const
{
    using STIdentity = SpectrumConverterContextIdentity;
    using MetaLightView = typename ML::template MetaLightView<STIdentity>;
    STIdentity stIdentity;

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
    MetaLightView metaLight = dMetaLights(stIdentity, lightIndex);
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
MRAY_HYBRID
Float DirectLightSamplerViewUniform<ML>::PdfLight(uint32_t index,
                                                  const MetaHit& hit,
                                                  const Ray& r) const
{
    // TODO............................
    Vector3 direction, position;

    using STIdentity = SpectrumConverterContextIdentity;
    using MetaLightView = typename ML::template MetaLightView<STIdentity>;
    STIdentity stIdentity;

    if(index >= dMetaLights.Size()) return Float(0);

    // Discrete sampling of such light (its uniform)
    uint32_t lightCount = dMetaLights.Size();
    Float lightCountF = Float(lightCount);
    Float selectionPDF = Float(1) / lightCountF;

    // Probability of sampling such direction from the particular light

    Float lightPDF = dMetaLights(stIdentity,index).PdfSolidAngle(hit, direction, position);

    return lightPDF * selectionPDF;
}

#include "LightSampler.hpp"