#pragma once

#include <cstdint>
#include "Core/Vector.h"

#include "Random.h"
#include "MetaLight.h"

struct LightSampleOutput
{
    uint32_t    lightIndex;
    Vector3     position;
    Spectrum    emission;

    MR_HF_DECL
    Pair<Ray, Vector2>  SampledRay(const Vector3& distantPoint) const;
};
using LightSample = SampleT<LightSampleOutput>;

// Not used atm
//template<class LSampler>
//concept LightSamplerC = requires(const LSampler& sampler,
//                                 RNGDispenser& rng,
//                                 const HitKeyPack& hitPack,
//                                 MetaHit metaHit)
//{
//    {sampler.SampleLight(rng, Vector3{})} -> std::same_as<LightSample>;
//    {sampler.PdfLight(uint32_t{}, metaHit, Ray{})} -> std::same_as<Float>;
//    {sampler.PdfLight(hitPack, metaHit, Ray{})} -> std::same_as<Float>;
//};

template <class MetaLightArray>
class DirectLightSamplerUniform
{
    using MetaLightArrayView = typename MetaLightArray::View;
    private:
    MetaLightArrayView  dMetaLights;
    LightLookupTable    dLightIndexTable;

    public:
    static constexpr uint32_t SampleLightRNCount = 1 + MetaLightArray::SampleSolidAngleRNCountWorst;

    public:
    // Constructors
    DirectLightSamplerUniform(const MetaLightArrayView&,
                              const LightLookupTable&);

    template <class SpectrumConverter>
    MR_HF_DECL
    LightSample SampleLight(RNGDispenser& rng,
                            const SpectrumConverter&,
                            const Vector3& distantPoint,
                            const Vector3& distantNormal,
                            const RayConeSurface& distantRayConeSurface) const;

    MR_HF_DECL
    Float       PdfLight(uint32_t index, const MetaHit& hit,
                         const Ray& r) const;
    MR_HF_DECL
    Float       PdfLight(const HitKeyPack&, const MetaHit& hit,
                         const Ray& r) const;
};

#include "LightSampler.hpp"