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
        normal = (*normalMapTex)(surface.uv, surface.dpdu, surface.dpdv).value();
        normal.NormalizeSelf();
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
    Vector3 normal = (normalMapTex)
        ? (*normalMapTex)(surface.uv, surface.dpdu, surface.dpdv).value()
        : Vector3::ZAxis();
    normal.NormalizeSelf();
    Float pdf = PDFCosDirection(wILocal, normal);
    return std::max(pdf, Float(0));
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LambertMaterial<ST>::Evaluate(const Ray& wI,
                                       const Vector3&,
                                       const Surface& surface) const
{
    Vector3 normal = (normalMapTex)
        ? (*normalMapTex)(surface.uv, surface.dpdu, surface.dpdv).value()
        : Vector3::ZAxis();
    normal.NormalizeSelf();
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

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float UnrealMaterial<ST>::MISRatio(Float metallic, Float specular,
                                   Float avgAlbedo) const
{
    // This function returns diffuse selection probability
    // Diffuse part
    Float integralDiffuse = Float(2) * MathConstants::Pi<Float>() * avgAlbedo;
    integralDiffuse *= (Float(1) - metallic);
    // Specular part
    Float specularRatio = Math::Lerp(specular, avgAlbedo, metallic);
    Float total = specularRatio + integralDiffuse;
    return (total == Float(0)) ? Float(0) : integralDiffuse / total;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float UnrealMaterial<ST>::ConvertProbHToL(Float VdH, Float pdfH) const
{
    // VNDFGGXSmithPDF returns sampling of H Vector
    // convert it to sampling probability of L Vector
    return (VdH == Float(0)) ? Float(0) : (pdfH / (Float(4) * VdH));
}

template <class ST>
Spectrum UnrealMaterial<ST>::CalculateF0(Spectrum albedo, Float metallic,
                                         Float specular) const
{
    // https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    // Utilizing proposed specular value to blend base color with specular parameter
    static constexpr float SpecularMax = 0.08f;
    specular *= SpecularMax;
    return Spectrum::Lerp(Spectrum(specular), albedo, metallic);
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Tuple<Float, Float, Float, Spectrum>
UnrealMaterial<ST>::FetchData(const Surface& s) const
{
    Float roughness = roughnessTex(s.uv, s.dpdu, s.dpdv).value();
    Float metallic = metallicTex(s.uv, s.dpdu, s.dpdv).value();
    Float specular = specularTex(s.uv, s.dpdu, s.dpdv).value();
    Spectrum albedo = albedoTex(s.uv, s.dpdu, s.dpdv).value();
    return Tuple{roughness, metallic, specular, albedo};
}

template <class SpectrumTransformer>
MRAY_HYBRID MRAY_CGPU_INLINE
UnrealMaterial<SpectrumTransformer>::UnrealMaterial(const SpectrumConverter& specTransformer,
                                                    const DataSoA& soa, MaterialKey mk)
    : albedoTex(specTransformer, soa.dAlbedo[mk.FetchIndexPortion()])
    , normalMapTex(soa.dNormalMaps[mk.FetchIndexPortion()])
    , roughnessTex(soa.dRoughness[mk.FetchIndexPortion()])
    , specularTex(soa.dSpecular[mk.FetchIndexPortion()])
    , metallicTex(soa.dMetallic[mk.FetchIndexPortion()])
    , mediumId(soa.dMediumIds[mk.FetchIndexPortion()])
{}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BxDFResult> UnrealMaterial<ST>::SampleBxDF(const Vector3& wO,
                                                   const Surface& surface,
                                                   RNGDispenser& dispenser) const
{
    static constexpr int DIFFUSE = 0;
    static constexpr int SPECULAR = 1;


    using namespace Distribution;
    auto [roughness, metallic, specular, albedo] = FetchData(surface);

    Float alpha = roughness * roughness;
    // TODO: This should be in spectrum converter, for RGB factor will
    // be 0.33..3, but for other it will be 0.25
    Float avgAlbedo = albedo.Sum() * Float(0.3333);

    // Get two random numbers
    Float sXI = dispenser.NextFloat<0>();
    Vector2 xi = dispenser.NextFloat2D<1>();
    // We do not use extra RNG for single samlple MIS
    // we just bisect the given samples
    // TODO: Is this correct?
    Float misRatio = MISRatio(metallic, specular, avgAlbedo);
    std::array<Float, 2> misWeights = {misRatio, Float(1) - misRatio};
    bool doDiffuseSample = (sXI < misRatio);

    // Microfacet dist functions are all in tangent space
    Quaternion toTangentSpace = surface.shadingTBN;
    if(normalMapTex)
    {
        Vector3 normal = (*normalMapTex)(surface.uv, surface.dpdu, surface.dpdv).value();
        normal.NormalizeSelf();
        // Due to normal change our direction sample should be aligned as well
        toTangentSpace = Quaternion::RotationBetweenZAxis(normal) * toTangentSpace;
    }
    // Bring wO all the way to the tangent space
    Vector3 V = toTangentSpace.ApplyRotation(wO);
    Vector3 L, H;
    std::array<Float, 2> pdf;
    if(doDiffuseSample)
    {
        auto s = Common::SampleCosDirection(xi);
        L = s.value;
        pdf[DIFFUSE] = s.pdf;
        //
        H = (L + V).Normalize();
        Float VdH = std::max(Float(0), V.Dot(H));
        pdf[SPECULAR] = BxDF::VNDFGGXSmithPDF(V, H, alpha);
        pdf[SPECULAR] = ConvertProbHToL(VdH, pdf[SPECULAR]);
    }
    else
    {
        auto s = BxDF::VNDFGGXSmithSample(V, alpha, xi);
        H = s.value;
        L = Graphics::Reflect(H, V);
        Float VdH = std::max(Float(0), V.Dot(H));
        pdf[SPECULAR] = ConvertProbHToL(VdH, s.pdf);
        //
        pdf[DIFFUSE] = Common::PDFCosDirection(L);
    }

    //=======================//
    //      Calculation      //
    //=======================//
    // Specular
    Float VdH = std::max(Float(0), V.Dot(H));
    Float LdH = std::max(Float(0), L.Dot(H));
    Float NdH = std::max(Float(0), Common::DotN(H));
    Float NdV = std::max(Float(0), Common::DotN(V));
    // Normal Distribution Function (GGX)
    Float D = BxDF::DGGX(NdH, alpha);
    // Shadowing Term (Smith Model)
    Float G = BxDF::GSmithCorralated(V, L, alpha);
    G = (LdH == Float(0)) ? Float(0) : G;
    G = (VdH == Float(0)) ? Float(0) : G;
    // Fresnel Term (Schlick's Approx)
    Spectrum f0 = CalculateF0(albedo, metallic, specular);
    Spectrum F = BxDF::FSchlick(VdH, f0);
    // Notice that NdL terms are canceled out
    Spectrum specularTerm = D * F * G * 0.25f / NdV;
    specularTerm = (NdV == Float(0)) ? Spectrum::Zero() : specularTerm;

    // Edge case D is unstable since alpha is too small
    // fall back to cancelled version
    if(isinf(D) ||  // alpha is small
       isnan(D))    // alpha is zero
    {
        specularTerm = G * F / BxDF::GSmithSingle(V, alpha);
        pdf[SPECULAR] = Float(1);
    }

    // Diffsue Portion
    // Blend between albedo<->black for metallic material
    Float NdL = std::max(0.0f, L[0]);
    Spectrum diffuseAlbedo = (Float(1) - metallic) * albedo;
    Spectrum diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi<Float>();
    diffuseTerm *= BxDF::BurleyDiffuseCorrection(NdL, NdV, LdH, roughness);
    Float pdfOut = MIS::BalanceCancelled<2>(pdf, misWeights);
    // Convert direction to local space
    L = toTangentSpace.ApplyInvRotation(L);

    Spectrum reflectance = diffuseTerm + specularTerm;
    // Go all the way to the local space
    return SampleT<BxDFResult>
    {
        .value = BxDFResult
        {
            .wI = Ray(L, surface.position),
            .reflectance = reflectance,
            .mediumKey = mediumId
        },
        .pdf = pdfOut
    };
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float UnrealMaterial<ST>::Pdf(const Ray& wI,
                              const Vector3& wO,
                              const Surface& surface) const
{
    using namespace Distribution;
    auto [roughness, metallic, specular, albedo] = FetchData(surface);
    Float alpha = roughness * roughness;
    // TODO: This should be in spectrum converter, for RGB factor will
    // be 0.33..3, but for other it will be 0.25
    Float avgAlbedo = albedo.Sum() * Float(0.3333);
    // Microfacet dist functions are all in tangent space
    Quaternion toTangentSpace = surface.shadingTBN;
    if(normalMapTex)
    {
        Vector3 normal = (*normalMapTex)(surface.uv, surface.dpdu, surface.dpdv).value();
        normal.NormalizeSelf();
        // Due to normal change our direction sample should be aligned as well
        toTangentSpace = Quaternion::RotationBetweenZAxis(normal) * toTangentSpace;
    }
    // Bring wO, wI all the way to the tangent space
    Vector3 V = toTangentSpace.ApplyRotation(wO);
    Vector3 L = toTangentSpace.ApplyRotation(wI.Dir());
    Vector3 H = (L + V).Normalize();
    //
    Float misRatio = MISRatio(metallic, specular, avgAlbedo);
    std::array<Float, 2> weights = {misRatio, Float(1) - misRatio};
    std::array<Float, 2> pdf;
    // Diffuse pdf
    pdf[0] = Common::PDFCosDirection(L);
    // Specular pdf
    Float NdH = std::max(Float(0), Common::DotN(H));
    Float VdH = std::max(Float(0), V.Dot(H));
    Float D = BxDF::DGGX(NdH, alpha);
    pdf[1] = BxDF::VNDFGGXSmithPDF(V, H, alpha);
    pdf[1] = (std::isnan(D) || std::isinf(D)) ? Float(0) : pdf[1];
    pdf[1] = (VdH == Float(0)) ? Float(0) : pdf[1] / (Float(4) * VdH);

    Float result = MIS::BalanceCancelled<2>(pdf, weights);
    return result;
}

template <class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum UnrealMaterial<ST>::Evaluate(const Ray& wI,
                                      const Vector3& wO,
                                      const Surface& surface) const
{
    using namespace Distribution;
    // Get the data first
    auto [roughness, metallic, specular, albedo] = FetchData(surface);
    Float alpha = roughness * roughness;
    // TODO: This should be in spectrum converter, for RGB factor will
    // be 0.33..3, but for other it will be 0.25
    Float avgAlbedo = albedo.Sum() * Float(0.3333);
    // Microfacet dist functions are all in tangent space
    Quaternion toTangentSpace = surface.shadingTBN;
    if(normalMapTex)
    {
        Vector3 normal = (*normalMapTex)(surface.uv, surface.dpdu, surface.dpdv).value();
        normal.NormalizeSelf();
        // Due to normal change our direction sample should be aligned as well
        toTangentSpace = Quaternion::RotationBetweenZAxis(normal) * toTangentSpace;
    }
    // Bring wO, wI all the way to the tangent space
    Vector3 V = toTangentSpace.ApplyRotation(wO);
    Vector3 L = toTangentSpace.ApplyRotation(wI.Dir());
    Vector3 H = (L + V).Normalize();

    //=======================//
    //   Calculate Specular  //
    //=======================//
    Float LdH = std::max(Float(0), L.Dot(H));
    Float VdH = std::max(Float(0), V.Dot(H));
    Float NdH = std::max(Float(0), Common::DotN(H));
    Float NdV = std::max(Float(0), Common::DotN(V));
    // Normal Distribution Function (GGX)
    Float D = BxDF::DGGX(NdH, alpha);
    // NDF could exceed the precision (alpha is small) or returns nan
    // (alpha is zero).
    // Assume geometry term was going to zero out the contribution
    // and zero here as well.
    D = (isnan(D) || isinf(D)) ? Float(0) : D;
    // Shadowing Term (Smith Model)
    Float G = BxDF::GSmithCorralated(V, L, alpha);
    G = (LdH == Float(0)) ? Float(0) : G;
    G = (VdH == Float(0)) ? Float(0) : G;
    // Fresnel Term (Schlick's Approx)
    Spectrum f0 = CalculateF0(albedo, metallic, specular);
    Spectrum F = BxDF::FSchlick(VdH, f0);
    // Notice that NdL terms are canceled out
    Spectrum specularTerm = D * F * G * Float(0.25) / NdV;
    specularTerm = (NdV == Float(0)) ? Spectrum::Zero() : specularTerm;

    //=======================//
    //   Calculate Diffuse   //
    //=======================//
    // Blend between albedo<->black for metallic material
    Float NdL = std::max(Float(0), Common::DotN(L));
    Spectrum diffuseAlbedo = (Float(1) - metallic) * albedo;
    Spectrum diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi<Float>();
    diffuseTerm *= BxDF::BurleyDiffuseCorrection(NdL, NdV, LdH, roughness);

    // All Done!
    return diffuseTerm + specularTerm;
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
    allResident &= roughnessTex(surface.uv, surface.dpdu,
                                surface.dpdv).has_value();
    allResident &= specularTex(surface.uv, surface.dpdu,
                               surface.dpdv).has_value();
    allResident &= metallicTex(surface.uv, surface.dpdu,
                               surface.dpdv).has_value();
    return allResident;
}
}