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
    auto [wO, pdf] = SampleCosDirection(xi);

    // Check normal Mapping
    Vector3 normal = Vector3::ZAxis();
    if(normalMapTex)
    {
        normal = (*normalMapTex)(surface.uv).value();
        // Due to normal change our direction sample should be aligned as well
        wO = Quaternion::RotationBetweenZAxis(normal).ApplyRotation(wO);
    }

    // Before transform calculate reflectance
    Float nDotL = fmax(normal.Dot(wO), Float{0});
    Spectrum albedo = albedoTex(surface.uv,
                                surface.dpdu,
                                surface.dpdv).value();
    Spectrum reflectance = albedo * nDotL * MathConstants::InvPi<Float>();

    // Material is responsible for transforming out of primitive's
    // shading space (same goes for wI but lambert material is
    // wI invariant so we did not convert it)
    wO = surface.shadingTBN.ApplyInvRotation(wO);
    // Lambert material is **not** asubsurface material,
    // directly delegate the incoming position as outgoing
    Ray wORay = Ray(wO, surface.position);

    return SampleT<BxDFResult>
    {
        .value = BxDFResult
        {
            .wO = wORay,
            .reflectance = reflectance,
            .mediumId = mediumId
        },
        .pdf = pdf
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LambertMaterial<ST>::Pdf(const Ray&,
                               const Ray& wO,
                               const Surface& surface) const
{
    using Distribution::Common::PDFCosDirection;

    Vector3 normal = (normalMapTex) ? (*normalMapTex)(surface.uv).value()
                                    : Vector3::ZAxis();
    Float pdf = PDFCosDirection(wO.Dir(), normal);
    return std::max(pdf, Float(0));
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LambertMaterial<ST>::Evaluate(const Ray& wO,
                                       const Vector3&,
                                       const Surface& surface) const
{
    Vector3 normal = (normalMapTex) ? (*normalMapTex)(surface.uv).value()
                                    : Vector3::ZAxis();

    // Calculate lightning in local space since
    // wO and wI is already in world space
    normal = surface.shadingTBN.ApplyInvRotation(normal);

    Float nDotL = std::max(normal.Dot(wO.Dir()), Float(0));
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
ReflectMaterial<ST>::ReflectMaterial(const SpectrumConverter& sTransContext,
                                     const DataSoA& soa, MaterialKey mk)
    // TODO: Add medium here later
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> ReflectMaterial<ST>::SampleBxDF(const Vector3& wI,
                                                    const Surface& surface,
                                                    RNGDispenser&) const
{
    // It is less operations to convert the normal to world space
    // since we will need to put wI to local space then convert wO
    // to world space.
    // TODO: Maybe fast reflect (that assumes normal is ZAxis) maybe faster?
    // Probably not.
    Vector3 normal = Vector3::ZAxis();
    Vector3 worldNormal = surface.shadingTBN.ApplyInvRotation(normal);
    Vector3 wO = Graphics::Reflect(worldNormal, wI);
    // Directly delegate position, this is not a subsurface material
    Ray wORay = Ray(wO, surface.position);
    return SampleT<BxDFResult>
    {
        .value = BxDFResult
        {
            .wO = wORay,
            .reflectance = Spectrum(1.0),
            // TODO: Change this later
            .mediumId = MediumKey::InvalidKey()
        },
        .pdf = Float(1.0)
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float ReflectMaterial<ST>::Pdf(const Ray& wI,
                               const Ray& wO,
                               const Surface& surface) const
{
    // We can not sample this
    return Float(0.0);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum ReflectMaterial<ST>::Evaluate(const Ray& wO,
                                       const Vector3& wI,
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
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> RefractMaterial<ST>::SampleBxDF(const Vector3& wI,
                                                    const Surface& surface,
                                                    RNGDispenser&) const
{

}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float RefractMaterial<ST>::Pdf(const Ray& wI,
                               const Ray& wO,
                               const Surface& surface) const
{
    // We can not sample this
    return Float(0.0);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum RefractMaterial<ST>::Evaluate(const Ray& wO,
                                       const Vector3& wI,
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