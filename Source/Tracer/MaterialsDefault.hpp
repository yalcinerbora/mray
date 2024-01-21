#pragma once

namespace LambertMatDetail
{

template <class SpectrumTransformer>
MRAY_HYBRID MRAY_CGPU_INLINE
LambertMaterial<SpectrumTransformer>::LambertMaterial(const typename SpectrumTransformer::Converter& specTransformer,
                                                      const LambertMatData& soa, MaterialId id)
    : albedoTex(specTransformer, soa.dAlbedo[id.FetchIndexPortion()])
    , normalMapTex(soa.dNormalMaps[id.FetchIndexPortion()])
    , mediumId(soa.dMediumIds[id.FetchIndexPortion()])
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> LambertMaterial<ST>::SampleBxDF(const Vector3& wI,
                                                   const Surface& surface,
                                                   RNGDispenser& dispenser) const
{
    using GraphicsFunctions::SampleCosDirection;
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
        .sampledResult = BxDFResult
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
Float LambertMaterial<ST>::Pdf(const Ray& wI,
                               const Ray& wO,
                               const Surface& surface) const
{
    using GraphicsFunctions::PDFCosDirection;

    Vector3 normal = (normalMapTex) ? (*normalMapTex)(surface.uv)
                                    : Vector3::ZAxis();
    Float pdf = PDFCosDirection(wO.Dir(), normal);
    return max(pdf, Float{0});
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LambertMaterial<ST>::SampleRNCount() const
{
    return 2;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LambertMaterial<ST>::Evaluate(const Ray& wO,
                                       const Vector3& wI,
                                       const Surface& surface) const
{
    Vector3 normal = (normalMapTex) ? (*normalMapTex)(surface.uv)
                                    : Vector3::ZAxis();

    // Calculate lightning in local space since
    // wO and wI is already in world space
    normal = surface.shadingTBN.ApplyInvRotation(normal);

    Float nDotL = max(normal.Dot(wO.Dir()), 0.0);
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
Spectrum LambertMaterial<ST>::Emit(const Vector3& wO, const Surface& surf) const
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
