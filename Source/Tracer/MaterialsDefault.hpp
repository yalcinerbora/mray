#pragma once

namespace LambertMatDetail
{

template <class SpectrumTransformer>
MRAY_HYBRID MRAY_CGPU_INLINE
LambertMaterial<SpectrumTransformer>::LambertMaterial(const SpectrumConverter& specTransformer,
                                                      const DataSoA& soa, MaterialKey mk)
    : albedoTex(specTransformer, soa.dAlbedo[mk.FetchIndexPortion()])
    , normalMapTex(soa.dNormalMaps[mk.FetchIndexPortion()])
    , mediumId(soa.dMediumIds[mk.FetchIndexPortion()])
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> LambertMaterial<ST>::SampleBxDF(const Vector3&,
                                                   const Surface& surface,
                                                   RNGDispenser& dispenser) const
{
    using Distribution::Common::SampleCosDirection;
    // Sampling a vector from cosine weighted hemispherical distribution
    Vector2 xi = dispenser.NextFloat2D<0>();
    auto [wI, pdf] = SampleCosDirection(xi);
    // Before lifting up to the local space calculate dot product
    Float nDotL = std::max(wI[2], Float{0});

    // Check normal Mapping
    Vector3 normal = Vector3::ZAxis();
    if(normalMapTex)
    {
        normal = (*normalMapTex)(surface.uv).value();
        // Due to normal change our direction sample should be aligned as well
        wI = Quaternion::RotationBetweenZAxis(normal).ApplyRotation(wI);
    }

    // Before transform calculate reflectance
    Spectrum albedo = albedoTex(surface.uv, surface.dpdu, surface.dpdv).value();
    Spectrum reflectance = albedo * nDotL * MathConstants::InvPi<Float>();

    // Material is responsible for transforming out of primitive's
    // shading space (same goes for wI but lambert material is
    // wI invariant so we did not convert it)
    wI = surface.shadingTBN.ApplyInvRotation(wI);
    // Lambert material is **not** asubsurface material,
    // directly delegate the incoming position as outgoing
    Ray wIRay = Ray(wI, surface.position);

    return SampleT<BxDFResult>
    {
        .value = BxDFResult
        {
            .wI = wIRay,
            .reflectance = reflectance,
            .mediumKey = mediumId
        },
        .pdf = pdf
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LambertMaterial<ST>::Pdf(const Ray& wI,
                               const Vector3&,
                               const Surface& surface) const
{
    using Distribution::Common::PDFCosDirection;
    Vector3 wILocal = surface.shadingTBN.ApplyRotation(wI.Dir());
    Vector3 normal = (normalMapTex) ? (*normalMapTex)(surface.uv).value()
                                    : Vector3::ZAxis();
    Float pdf = PDFCosDirection(wILocal, normal);
    return std::max(pdf, Float(0));
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LambertMaterial<ST>::Evaluate(const Ray& wI,
                                       const Vector3&,
                                       const Surface& surface) const
{
    Vector3 normal = (normalMapTex) ? (*normalMapTex)(surface.uv).value()
                                    : Vector3::ZAxis();
    // Calculate lightning in local space since
    // wO and wI is already in local space
    Vector3 wILocal = surface.shadingTBN.ApplyRotation(wI.Dir());

    Float nDotL = std::max(normal.Dot(wILocal), Float(0));
    Spectrum albedo = sTransContext(albedoTex(surface.uv,
                                              surface.dpdu,
                                              surface.dpdv));
    return nDotL * albedo * MathConstants::InvPi<Float>();
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool LambertMaterial<ST>::IsEmissive() const
{
    return false;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LambertMaterial<ST>::Emit(const Vector3&, const Surface&) const
{
    return Spectrum::Zero();
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool LambertMaterial<ST>::IsAllTexturesAreResident(const Surface& surface) const
{
    bool allResident = true;
    allResident &= albedoTex(surface.uv, surface.dpdu,
                             surface.dpdv).has_value();
    if(normalMapTex)
    {
        allResident &= (*normalMapTex)(surface.uv).has_value();
    }
    return allResident;
}

}

namespace ReflectMatDetail
{

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
ReflectMaterial<ST>::ReflectMaterial(const SpectrumConverter&,
                                     const DataSoA&, MaterialKey)
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> ReflectMaterial<ST>::SampleBxDF(const Vector3& wO,
                                                    const Surface& surface,
                                                    RNGDispenser&) const
{
    // It is less operations to convert the normal to local space
    // since we will need to put wI to local space then convert wO
    // to local space.
    // TODO: Maybe fast reflect (that assumes normal is ZAxis) maybe faster?
    Vector3 normal = Vector3::ZAxis();
    Vector3 localNormal = surface.shadingTBN.ApplyInvRotation(normal);
    Vector3 wI = Graphics::Reflect(localNormal, wO);
    // Directly delegate position, this is not a subsurface material
    Ray wIRay = Ray(wI, surface.position);
    return SampleT<BxDFResult>
    {
        .value = BxDFResult
        {
            .wI = wIRay,
            .reflectance = Spectrum(1.0),
            // TODO: Change this later
            .mediumKey = MediumKey::InvalidKey()
        },
        .pdf = Float(1.0)
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float ReflectMaterial<ST>::Pdf(const Ray& wI,
                               const Vector3& wO,
                               const Surface& surface) const
{
    // We can not sample this
    return Float(0.0);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum ReflectMaterial<ST>::Evaluate(const Ray& wI,
                                       const Vector3& wO,
                                       const Surface& surface) const
{
    return Spectrum(1);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool ReflectMaterial<ST>::IsEmissive() const
{
    return false;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum ReflectMaterial<ST>::Emit(const Vector3& wO,
                                   const Surface& surf) const
{
    return Spectrum::Zero();
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool ReflectMaterial<ST>::IsAllTexturesAreResident(const Surface& surface) const
{
    return true;
}

}

namespace RefractMatDetail
{

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
RefractMaterial<ST>::RefractMaterial(const SpectrumConverter& sTransContext,
                                     const DataSoA& soa, MaterialKey mk)
    // TODO: Add medium here later
    : mKeyIn(soa.dMediumIds[mk.FetchIndexPortion()].first)
    , mKeyOut(soa.dMediumIds[mk.FetchIndexPortion()].second)
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> RefractMaterial<ST>::SampleBxDF(const Vector3& wO,
                                                    const Surface& surface,
                                                    RNGDispenser& rng) const
{
    // Fetch Mat
    // Determine medium index of refractions
    //uint32_t mediumIndex = matData.mediumIndices[matId];
    //Float iIOR = matData.dMediums[mediumIndex]->IOR();
    //Float dIOR = matData.dMediums[matData.baseMediumIndex]->IOR();
    //TODO:!!!
    Float fromEta = Float(1);
    Float toEta = Float(1.5);
    MediumKey fromMedium;
    MediumKey toMedium;

    // Check if we are exiting or entering
    bool entering = !surface.backSide;
    if(!entering)
    {
        std::swap(fromEta, toEta);
        std::swap(fromMedium, toMedium);
    }

    // Surface is aligned with the ray (N dot Dir is always positive)
    const Vector3 nLocal = surface.shadingTBN.ApplyInvRotation(Vector3::ZAxis());
    Float cosTheta = std::abs(wO.Dot(nLocal));

    // Calculate Fresnel Term
    Float f = Distribution::BxDF::FresnelDielectric(cosTheta, fromEta, toEta);

    // Sample ray according to the Fresnel term
    Float xi = rng.NextFloat<0>();
    bool doReflection = (xi < f);

    Vector3 wI = (doReflection)
        ? Graphics::Reflect(nLocal, wO)
        // Since we refract via fresnel, total internal reflection
        // should not happen
        : Graphics::Refract(nLocal, wO, fromEta, toEta).value();
    //
    MediumKey outMedium = (doReflection) ? fromMedium : toMedium;
    Float pdf           = (doReflection) ? f : (Float(1) - f);
    //
    return SampleT<BxDFResult>
    {
        .value = BxDFResult
        {
            .wI = Ray(wI, surface.position).Nudge(nLocal),
            .reflectance = Spectrum(pdf),
            .mediumKey = outMedium
        },
        .pdf = pdf
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float RefractMaterial<ST>::Pdf(const Ray& wI,
                               const Vector3& wO,
                               const Surface& surface) const
{
    // We can not sample this
    return Float(0.0);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum RefractMaterial<ST>::Evaluate(const Ray& wI,
                                       const Vector3& wO,
                                       const Surface& surface) const
{
    return Spectrum(1);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool RefractMaterial<ST>::IsEmissive() const
{
    return false;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum RefractMaterial<ST>::Emit(const Vector3& wO,
                                   const Surface& surf) const
{
    return Spectrum::Zero();
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool RefractMaterial<ST>::IsAllTexturesAreResident(const Surface& surface) const
{
    return true;
}

}

namespace UnrealMatDetail
{
template <class SpectrumTransformer>
MRAY_HYBRID MRAY_CGPU_INLINE
UnrealMaterial<SpectrumTransformer>::UnrealMaterial(const SpectrumConverter& specTransformer,
                                                    const DataSoA& soa, MaterialKey mk)
    : albedoTex(specTransformer, soa.dAlbedo[mk.FetchIndexPortion()])
    , normalMapTex(soa.dNormalMaps[mk.FetchIndexPortion()])
    , roughness(soa.dRoughness[mk.FetchIndexPortion()])
    , specular(soa.dSpecular[mk.FetchIndexPortion()])
    , metallic(soa.dMetallic[mk.FetchIndexPortion()])
    , mediumId(soa.dMediumIds[mk.FetchIndexPortion()])
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> UnrealMaterial<ST>::SampleBxDF(const Vector3& wO,
                                                   const Surface& surface,
                                                   RNGDispenser& dispenser) const
{

    return SampleT<BxDFResult> {};
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float UnrealMaterial<ST>::Pdf(const Ray& wI,
                              const Vector3& wO,
                              const Surface& surface) const
{
    return 0;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum UnrealMaterial<ST>::Evaluate(const Ray& wI,
                                      const Vector3& wO,
                                      const Surface& surface) const
{
    return Spectrum::Zero();
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool UnrealMaterial<ST>::IsEmissive() const
{
    return false;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum UnrealMaterial<ST>::Emit(const Vector3&, const Surface&) const
{
    return Spectrum::Zero();
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool UnrealMaterial<ST>::IsAllTexturesAreResident(const Surface& surface) const
{
    bool allResident = true;
    allResident &= albedoTex(surface.uv, surface.dpdu,
                             surface.dpdv).has_value();
    if(normalMapTex)
    {
        allResident &= (*normalMapTex)(surface.uv).has_value();
    }
    allResident &= roughness(surface.uv, surface.dpdu,
                             surface.dpdv).has_value();
    allResident &= specular(surface.uv, surface.dpdu,
                            surface.dpdv).has_value();
    allResident &= metallic(surface.uv, surface.dpdu,
                            surface.dpdv).has_value();
    return allResident;
}
}