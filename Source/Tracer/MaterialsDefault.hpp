#pragma once

#include "MaterialsDefault.h"
#include "DistributionFunctions.h"

namespace LambertMatDetail
{

template <class SC>
MR_GF_DEF
NormalMap LambertMaterial<SC>::GetNormalMap(const DataSoA& soa, MaterialKey mk)
{
    return soa.dNormalMaps[mk.FetchIndexPortion()];
}

template <class SpectrumContext>
MR_GF_DEF
LambertMaterial<SpectrumContext>::LambertMaterial(const SpectrumConverter& specTransformer,
                                                  const Surface& surface,
                                                  const DataSoA& soa, MaterialKey mk)
    : surface(surface)
{
    const auto albedoTex = AlbedoMap(specTransformer, soa.dAlbedo[mk.FetchIndexPortion()]);
    albedo = albedoTex(surface.uv, surface.dpdx, surface.dpdy);
}

template <class SC>
MR_GF_DEF
BxDFSample LambertMaterial<SC>::SampleBxDF(const Vector3&,
                                           RNGDispenser& dispenser) const
{
    using Distribution::Common::SampleCosDirection;
    // Sampling a vector from cosine weighted hemispherical distribution
    Vector2 xi = dispenser.NextFloat2D<0>();
    auto [wI, pdf] = SampleCosDirection(xi);
    // Before lifting up to the local space calculate dot product
    Float nDotL = Math::Max(wI[2], Float{0});

    // Check normal Mapping
    Quaternion toTangentSpace = surface.shadingTBN;
    // Before transform calculate reflectance
    Spectrum reflectance = albedo * nDotL * MathConstants::InvPi<Float>();

    // Material is responsible for transforming out of primitive's
    // shading space (same goes for wI but lambert material is
    // wI invariant so we did not convert it)
    wI = toTangentSpace.ApplyInvRotation(wI);
    // Lambert material is **not** a subsurface material,
    // directly delegate the incoming position as outgoing
    Ray wIRay = Ray(wI, surface.position);

    return BxDFSample
    {
        .wI = wIRay,
        .pdf = pdf,
        .eval = BxDFEval
        {
            .reflectance = reflectance,
        }
    };
}

template <class SC>
MR_GF_DEF
Float LambertMaterial<SC>::Pdf(const Ray& wI, const Vector3&) const
{
    using Distribution::Common::PDFCosDirection;
    Vector3 wILocal = surface.shadingTBN.ApplyRotation(wI.dir);
    Float pdf = PDFCosDirection(wILocal);
    return Math::Max(pdf, Float(0));
}

template <class SC>
MR_GF_DEF
BxDFEval LambertMaterial<SC>::Evaluate(const Ray& wI, const Vector3&) const
{
    // Check normal Mapping
    Quaternion toTangentSpace = surface.shadingTBN;
    // Calculate lightning tangent space
    Vector3 wILocal = toTangentSpace.ApplyRotation(wI.dir);
    Float nDotL = Math::Max(wILocal[2], Float(0));
    return BxDFEval
    {
        .reflectance = nDotL * albedo * MathConstants::InvPi<Float>(),
    };
}

template <class SC>
MR_GF_DEF
bool LambertMaterial<SC>::IsEmissive() const
{
    return false;
}

template <class SC>
MR_GF_DEF
Spectrum LambertMaterial<SC>::Emit(const Vector3&) const
{
    return Spectrum::Zero();
}

template <class SC>
MR_GF_DEF
Float LambertMaterial<SC>::Specularity() const
{
    return Float(0);
}

template <class SC>
MR_GF_DEF
RayConeSurface LambertMaterial<SC>::RefractRayCone(const RayConeSurface& r,
                                                   const Vector3&) const
{
    return r;
}

template <class SC>
MR_GF_DEF
bool LambertMaterial<SC>::IsAllTexturesAreResident(const Surface& s, const DataSoA& soa,
                                                   MaterialKey mk)
{
    const auto& albedoTex = soa.dAlbedo[mk.FetchIndexPortion()];
    const auto& normalTex = soa.dNormalMaps[mk.FetchIndexPortion()];

    bool allResident = albedoTex.IsResident(s.uv, s.dpdx, s.dpdy);
    if(normalTex)
        allResident &= normalTex->IsResident(s.uv, s.dpdx, s.dpdy);
    return allResident;
}

}

namespace ReflectMatDetail
{

template <class SC>
MR_GF_DEF
NormalMap ReflectMaterial<SC>::GetNormalMap(const DataSoA&, MaterialKey)
{
    return std::nullopt;
}

template <class SC>
MR_GF_DEF
ReflectMaterial<SC>::ReflectMaterial(const SpectrumConverter&,
                                     const Surface& surface,
                                     const DataSoA&, MaterialKey)
    : surface(surface)
{}

template <class SC>
MR_GF_DEF
BxDFSample ReflectMaterial<SC>::SampleBxDF(const Vector3& wO,
                                           RNGDispenser&) const
{
    // It is less operations to convert the normal to local space
    // since we will need to put wI to local space then convert wO
    // to local space.
    // TODO: Maybe fast reflect (that assumes normal is ZAxis) maybe faster?
    Vector3 localNormal = surface.shadingTBN.OrthoBasisZ();
    Vector3 wI = Graphics::Reflect(localNormal, wO);
    // Directly delegate position, this is not a subsurface material
    Ray wIRay = Ray(Math::Normalize(wI), surface.position);
    return BxDFSample
    {
        .wI = wIRay,
        .pdf = Float(1.0),
        .eval = BxDFEval
        {
            .reflectance = Spectrum(1.0),
        }
    };
}

template <class SC>
MR_GF_DEF
Float ReflectMaterial<SC>::Pdf(const Ray&, const Vector3&) const
{
    // We can not sample this
    return Float(0);
}

template <class SC>
MR_GF_DEF
BxDFEval ReflectMaterial<SC>::Evaluate(const Ray&, const Vector3&) const
{
    return BxDFEval
    {
        .reflectance = Spectrum(1)
    };
}

template <class SC>
MR_GF_DEF
bool ReflectMaterial<SC>::IsEmissive() const
{
    return false;
}

template <class SC>
MR_GF_DEF
Spectrum ReflectMaterial<SC>::Emit(const Vector3&) const
{
    return Spectrum::Zero();
}

template <class SC>
MR_GF_DEF
Float ReflectMaterial<SC>::Specularity() const
{
    return Float(1);
}

template <class SC>
MR_GF_DEF
RayConeSurface ReflectMaterial<SC>::RefractRayCone(const RayConeSurface& r,
                                                   const Vector3&) const
{
    return r;
}

template <class SC>
MR_GF_DEF
bool ReflectMaterial<SC>::IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                   MaterialKey)
{
    return true;
}

}

namespace RefractMatDetail
{

template <class SC>
MR_GF_DEF
NormalMap RefractMaterial<SC>::GetNormalMap(const DataSoA&, MaterialKey)
{
    return std::nullopt;
}

template <class SC>
MR_GF_DEF
RefractMaterial<SC>::RefractMaterial(const SpectrumConverter& sTransContext,
                                     const Surface& surface,
                                     const DataSoA& soa, MaterialKey mk)
    : surface(surface)
{
    // Fetch ior
    auto CoeffsToIoR = [&](Vector3 coeffs)
    {
        using namespace Distribution::Medium;
        return WavelengthToIoRCauchy(sTransContext.Wavelengths()[0], coeffs);
    };

    if constexpr(!SpectrumConverter::IsRGB)
    {
        frontIoR = CoeffsToIoR(soa.dFrontCauchyCoeffs[mk.FetchIndexPortion()]);
        backIoR = CoeffsToIoR(soa.dBackCauchyCoeffs[mk.FetchIndexPortion()]);
    }
    else
    {
        frontIoR = soa.dFrontCauchyCoeffs[mk.FetchIndexPortion()][0];
        backIoR = soa.dBackCauchyCoeffs[mk.FetchIndexPortion()][0];
    }
}

template <class SC>
MR_GF_DEF
BxDFSample RefractMaterial<SC>::SampleBxDF(const Vector3& wO,
                                           RNGDispenser& rng) const
{
    Float fromEta = frontIoR;
    Float toEta = backIoR;
    // Check if we are exiting or entering
    bool entering = !surface.backSide;
    if(!entering) std::swap(fromEta, toEta);

    // Surface is aligned with the ray (N dot Dir is always positive)
    const Vector3 nLocal = surface.shadingTBN.OrthoBasisZ();
    Float cosTheta = Math::Abs(Math::Dot(wO, nLocal));

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
    Float pdf = (doReflection) ? f : (Float(1) - f);
    //
    return BxDFSample
    {
        .wI = Ray(wI, surface.position).Nudge(nLocal),
        .pdf = pdf,
        .eval = BxDFEval
        {
            .reflectance     = Spectrum(pdf),
            .isPassedThrough = !doReflection,
            .isDispersed     = !doReflection
        }
    };
}

template <class SC>
MR_GF_DEF
Float RefractMaterial<SC>::Pdf(const Ray&, const Vector3&) const
{
    // We can not sample this
    return Float(0);
}

template <class SC>
MR_GF_DEF
BxDFEval RefractMaterial<SC>::Evaluate(const Ray& wI, const Vector3&) const
{
    bool isFront = Math::Dot(surface.geoNormal, wI.dir) > Float(0);
    return BxDFEval
    {
        .reflectance     = Spectrum(1),
        .isPassedThrough = !isFront,
        .isDispersed     = !isFront
    };
}

template <class SC>
MR_GF_DEF
bool RefractMaterial<SC>::IsEmissive() const
{
    return false;
}

template <class SC>
MR_GF_DEF
Spectrum RefractMaterial<SC>::Emit(const Vector3&) const
{
    return Spectrum::Zero();
}

template <class SC>
MR_GF_DEF
Float RefractMaterial<SC>::Specularity() const
{
    return Float(1);
}

template <class SC>
MR_GF_DEF
RayConeSurface RefractMaterial<SC>::RefractRayCone(const RayConeSurface& rayConeSurfIn,
                                                   const Vector3& wO) const
{
    auto Rotate2D_UL = [](Vector2 v, Float alpha) -> std::array<Vector2, 2>
    {
        // Matrix
        // [cos, -sin]
        // [sin, cos]
        //
        // Optimizations,
        //  cos(x) = cos(-x) (where x in [-90, 90])
        // -sin(x) = sin(-x) (where x in [-90, 90])
        // Thus we can rotate +- by single "cos and sin"
        const auto& [s, c] = Math::SinCos(alpha);
        return
        {
            Vector2(v[0] * c - v[1] * s,
                    v[0] * s + v[1] * c),
                    //
            Vector2(v[0] *  c + v[1] * s,
                    v[0] * -s + v[1] * c)
        };
    };

    auto Refract2D = [](Vector2 v, Vector2 n, Float fromEta, Float toEta) -> Vector2
    {
        // Just call 3D with zero literal
        // Pray compiler optimizes it (probably not since it is float)
        auto result = Graphics::Refract(Vector3(n, 0), Vector3(-v, 0),
                                        fromEta, toEta);
        return (result.has_value())
                ? Vector2(result.value())
                : Math::Normalize(v - n * Math::Dot(n, v));
    };

    Float fromEta = frontIoR;
    Float toEta = backIoR;
    // Swap eta if backside
    if(surface.backSide) std::swap(fromEta, toEta);

    // Refract the wO
    auto wI = Graphics::Refract(surface.geoNormal, wO, fromEta, toEta);
    // No change if reflection occurs.
    if(!wI.has_value()) return rayConeSurfIn;

    // From the RT Gems II Chapter 10 Figure 10-5.
    // This implementation follows the Falcor implementation.
    Vector3 d3D = -wO, t3D = wI.value();
    // Define the 2D space
    Vector3 x = Graphics::GSOrthonormalize(d3D, surface.geoNormal);
    Vector3 y = surface.geoNormal;

    // Project on the XY plane / Align the basis
    using Math::Dot;
    Vector2 d = Vector2(Dot(x, d3D), Dot(y, d3D));
    Vector2 t = Vector2(Dot(x, t3D), Dot(y, t3D));

    // BetaN is pre-added to the cones
    Float aperture = rayConeSurfIn.rayConeFront.aperture;
    Float coneWidth = rayConeSurfIn.rayConeFront.width;
    Float wSign = (coneWidth > Float(0)) ? Float(1) : Float(0);
    auto [du, dl] = Rotate2D_UL(d, wSign * aperture * Float(0.5));
    // We calculated all, now find the distance between two vectors
    // To calculate widths, we need to find the horizon hit points
    Vector2 orthoD = Vector2(-d[1], d[0]) * coneWidth * Float(0.5);
    Float uHitX = +orthoD[0] + du[0] * (-orthoD[1] / du[1]);
    Float lHitX = -orthoD[0] + dl[0] * (+orthoD[1] / dl[1]);

    // Normal spread calculation
    Float nSign = (uHitX > lHitX) ? Float(1) : Float(-1);
    Float deltaNTheta = -rayConeSurfIn.betaN * nSign * Float(0.5);
    auto [nu, nl] = Rotate2D_UL(Vector2(0, 1), deltaNTheta);
    Vector2 tu = Refract2D(du, nu, fromEta, toEta);
    Vector2 tl = Refract2D(dl, nl, fromEta, toEta);

    orthoD = Vector2(-d[1], d[0]);
    // **********
    // TODO: Report GCC-13 bug, this destroys constructor deduction
    // of Vector2(Args...) "where args are constrained with std::convertible_to"
    // of some? subsequent constructor usage
    //
    // Float wl = -uHitX * tu[1] / Dot(orthoD, Vector2(-tu[1], tu[0]));
    // Float wu = +lHitX * tl[1] / Dot(orthoD, Vector2(-tl[1], tl[0]));
    // **********
    // Trying to circumvent the trigger (it didn't take too long)
    Float wl = -uHitX * tu[1];
    wl /= Dot(orthoD, Vector2(-tu[1], tu[0]));
    Float wu = +lHitX * tl[1];
    wu /= Dot(orthoD, Vector2(-tl[1], tl[0]));

    // Calculate the new cone
    Float sign = Math::SignPM1(tu[0] * tl[1] - tu[1] * tl[0]);
    Float cosTheta = Math::Clamp(Dot(tu, tl), Float(-1), Float(1));
    Float newConeAperture = Math::ArcCos(cosTheta) * sign;
    newConeAperture = Math::Max(newConeAperture, MathConstants::Epsilon<Float>());

    Float width = wu + wl;
    RayConeSurface result = rayConeSurfIn;
    // We save the state **as if** it is pre-refracted (thus adding betaN,
    // "ConeAfterScatter" function will subtract it back
    assert(Math::IsFinite(newConeAperture));
    assert(newConeAperture != Float(0));
    result.rayConeBack = RayCone
    {
        .aperture = newConeAperture + rayConeSurfIn.betaN,
        .width = width
    };
    return result;
}

template <class SC>
MR_GF_DEF
bool RefractMaterial<SC>::IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                   MaterialKey)
{
    return true;
}

}

namespace UnrealMatDetail
{

template <class SC>
MR_GF_DEF
Float UnrealMaterial<SC>::MISRatio(Float avgAlbedo) const
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

template <class SC>
MR_GF_DEF
Float UnrealMaterial<SC>::ConvertProbHToL(Float VdH, Float pdfH) const
{
    // VNDFGGXSmithPDF returns sampling of H Vector
    // convert it to sampling probability of L Vector
    return (VdH == Float(0)) ? Float(0) : (pdfH / (Float(4) * VdH));
}

template <class SC>
MR_GF_DEF
Spectrum UnrealMaterial<SC>::CalculateF0() const
{
    // https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    // Utilizing proposed specular value to blend base color with specular parameter
    static constexpr float SpecularMax = 0.08f;
    Float specOut = specular * SpecularMax;
    return Math::Lerp(Spectrum(specOut), albedo, metallic);
}

template <class SC>
MR_GF_DEF
NormalMap UnrealMaterial<SC>::GetNormalMap(const DataSoA& soa, MaterialKey mk)
{
    return soa.dNormalMaps[mk.FetchIndexPortion()];
}

template <class SC>
MR_GF_DEF
UnrealMaterial<SC>::UnrealMaterial(const SpectrumConverter& specTransformer,
                                   const Surface& surface,
                                   const DataSoA& soa, MaterialKey mk)
    : surface(surface)
{
    auto albedoTex = AlbedoMap(specTransformer, soa.dAlbedo[mk.FetchIndexPortion()]);
    const auto& roughnessTex = soa.dRoughness[mk.FetchIndexPortion()];
    const auto& specularTex = soa.dSpecular[mk.FetchIndexPortion()];
    const auto& metallicTex = soa.dMetallic[mk.FetchIndexPortion()];
    roughness = roughnessTex(surface.uv, surface.dpdx, surface.dpdy);
    metallic = metallicTex(surface.uv, surface.dpdx, surface.dpdy);
    specular = specularTex(surface.uv, surface.dpdx, surface.dpdy);
    albedo = albedoTex(surface.uv, surface.dpdx, surface.dpdy);
}

template <class SC>
MR_GF_DEF
BxDFSample UnrealMaterial<SC>::SampleBxDF(const Vector3& wO,
                                          RNGDispenser& dispenser) const
{
    static constexpr int DIFFUSE = 0;
    static constexpr int SPECULAR = 1;
    using namespace Distribution;

    Float alpha = roughness * roughness;
    // TODO: This should be in spectrum converter, for RGB factor will
    // be 0.33..3, but for other it will be 0.25
    Float avgAlbedo = albedo.Sum() * Float(0.3333);

    // Get two random numbers
    Float sXi = dispenser.NextFloat<0>();
    Vector2 xi = dispenser.NextFloat2D<1>();
    // We do not use extra RNG for single sample MIS
    // we just bisect the given samples
    // TODO: Is this correct?
    Float misRatio = MISRatio(avgAlbedo);
    std::array<Float, 2> misWeights = {misRatio, Float(1) - misRatio};
    bool doDiffuseSample = (sXi < misRatio);

    // Microfacet dist functions are all in tangent space
    Quaternion toTangentSpace = surface.shadingTBN;
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
        H = Math::Normalize(L + V);
        Float VdH = Math::Max(Float(0), Math::Dot(V, H));
        pdf[SPECULAR] = BxDF::VNDFGGXSmithPDF(V, H, alpha);
        pdf[SPECULAR] = ConvertProbHToL(VdH, pdf[SPECULAR]);
    }
    else
    {
        auto s = BxDF::VNDFGGXSmithSample(V, alpha, xi);
        H = s.value;
        L = Graphics::Reflect(H, V);
        Float VdH = Math::Max(Float(0), Math::Dot(V, H));
        pdf[SPECULAR] = ConvertProbHToL(VdH, s.pdf);
        //
        pdf[DIFFUSE] = Common::PDFCosDirection(L);
    }

    //=======================//
    //      Calculation      //
    //=======================//
    // Specular
    Float VdH = Math::Max(Float(0), Math::Dot(V, H));
    Float LdH = Math::Max(Float(0), Math::Dot(L, H));
    Float NdH = Math::Max(Float(0), Common::DotN(H));
    Float NdV = Math::Max(Float(0), Common::DotN(V));
    // Normal Distribution Function (GGX)
    Float D = BxDF::DGGX(NdH, alpha);
    // Shadowing Term (Smith Model)
    Float G = BxDF::GSmithCorrelated(V, L, alpha);
    G = (LdH == Float(0)) ? Float(0) : G;
    G = (VdH == Float(0)) ? Float(0) : G;
    // Fresnel Term (Schlick's Approx)
    Spectrum f0 = CalculateF0();
    Spectrum F = BxDF::FSchlick(VdH, f0);
    // Notice that NdL terms are canceled out
    Spectrum specularTerm = D * F * G * 0.25f / NdV;
    specularTerm = (NdV == Float(0)) ? Spectrum::Zero() : specularTerm;

    // Edge case D is unstable since alpha is too small
    // fall back to cancelled version
    if(Math::IsInf(D) ||  // alpha is small
       Math::IsNaN(D))    // alpha is zero
    {
        specularTerm = G * F / BxDF::GSmithSingle(V, alpha);
        pdf[SPECULAR] = Float(1);
    }

    // Diffsue Portion
    // Blend between albedo<->black for metallic material
    Float NdL = Math::Max(Float(0), Common::DotN(L));
    Spectrum diffuseAlbedo = (Float(1) - metallic) * albedo;
    Spectrum diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi<Float>();
    diffuseTerm *= BxDF::BurleyDiffuseCorrection(NdL, NdV, LdH, roughness);
    Float pdfOut = MIS::BalanceCancelled<2>(pdf, misWeights);
    // Convert direction to local space
    L = toTangentSpace.ApplyInvRotation(L);

    Spectrum reflectance = diffuseTerm + specularTerm;
    // All done!
    return BxDFSample
    {
        .wI = Ray(L, surface.position),
        .pdf = pdfOut,
        .eval = BxDFEval
        {
            .reflectance = reflectance,
        }
    };
}

template <class SC>
MR_GF_DEF
Float UnrealMaterial<SC>::Pdf(const Ray& wI, const Vector3& wO) const
{
    using namespace Distribution;
    Float alpha = roughness * roughness;
    // TODO: This should be in spectrum converter, for RGB factor will
    // be 0.33..3, but for other it will be 0.25
    Float avgAlbedo = albedo.Sum() * Float(0.3333);
    // Microfacet dist functions are all in tangent space
    Quaternion toTangentSpace = surface.shadingTBN;
    // Bring wO, wI all the way to the tangent space
    Vector3 V = toTangentSpace.ApplyRotation(wO);
    Vector3 L = toTangentSpace.ApplyRotation(wI.dir);
    Vector3 H = Math::Normalize(L + V);
    //
    Float misRatio = MISRatio(avgAlbedo);
    std::array<Float, 2> weights = {misRatio, Float(1) - misRatio};
    std::array<Float, 2> pdf;
    // Diffuse pdf
    pdf[0] = Common::PDFCosDirection(L);
    // Specular pdf
    Float NdH = Math::Max(Float(0), Common::DotN(H));
    Float VdH = Math::Max(Float(0), Math::Dot(V, H));
    Float D = BxDF::DGGX(NdH, alpha);
    pdf[1] = BxDF::VNDFGGXSmithPDF(V, H, alpha);
    pdf[1] = (Math::IsNaN(D) || Math::IsInf(D)) ? Float(0) : pdf[1];
    pdf[1] = (VdH == Float(0)) ? Float(0) : pdf[1] / (Float(4) * VdH);

    Float result = MIS::BalanceCancelled<2>(pdf, weights);
    return result;
}

template <class SC>
MR_GF_DEF
BxDFEval UnrealMaterial<SC>::Evaluate(const Ray& wI, const Vector3& wO) const
{
    using namespace Distribution;
    Float alpha = roughness * roughness;
    // Microfacet dist functions are all in tangent space
    Quaternion toTangentSpace = surface.shadingTBN;
    // Bring wO, wI all the way to the tangent space
    Vector3 V = toTangentSpace.ApplyRotation(wO);
    Vector3 L = toTangentSpace.ApplyRotation(wI.dir);
    Vector3 H = Math::Normalize(L + V);

    //=======================//
    //   Calculate Specular  //
    //=======================//
    Float LdH = Math::Max(Float(0), Math::Dot(L, H));
    Float VdH = Math::Max(Float(0), Math::Dot(V, H));
    Float NdH = Math::Max(Float(0), Common::DotN(H));
    Float NdV = Math::Max(Float(0), Common::DotN(V));
    // Normal Distribution Function (GGX)
    Float D = BxDF::DGGX(NdH, alpha);
    // NDF could exceed the precision (alpha is small) or returns nan
    // (alpha is zero).
    // Assume geometry term was going to zero out the contribution
    // and zero here as well.
    D = (Math::IsNaN(D) || Math::IsInf(D)) ? Float(0) : D;
    // Shadowing Term (Smith Model)
    Float G = BxDF::GSmithCorrelated(V, L, alpha);
    G = (LdH == Float(0)) ? Float(0) : G;
    G = (VdH == Float(0)) ? Float(0) : G;
    // Fresnel Term (Schlick's Approx)
    Spectrum f0 = CalculateF0();
    Spectrum F = BxDF::FSchlick(VdH, f0);
    // Notice that NdL terms are canceled out
    Spectrum specularTerm = D * F * G * Float(0.25) / NdV;
    specularTerm = (NdV == Float(0)) ? Spectrum::Zero() : specularTerm;

    //=======================//
    //   Calculate Diffuse   //
    //=======================//
    // Blend between albedo<->black for metallic material
    Float NdL = Math::Max(Float(0), Common::DotN(L));
    Spectrum diffuseAlbedo = (Float(1) - metallic) * albedo;
    Spectrum diffuseTerm = NdL * diffuseAlbedo * MathConstants::InvPi<Float>();
    diffuseTerm *= BxDF::BurleyDiffuseCorrection(NdL, NdV, LdH, roughness);

    // All Done!
    return BxDFEval
    {
        .reflectance = diffuseTerm + specularTerm
    };
}

template <class SC>
MR_GF_DEF
bool UnrealMaterial<SC>::IsEmissive() const
{
    return false;
}

template <class SC>
MR_GF_DEF
Spectrum UnrealMaterial<SC>::Emit(const Vector3&) const
{
    return Spectrum::Zero();
}

template <class SC>
MR_GF_DEF
Float UnrealMaterial<SC>::Specularity() const
{
    // TODO: This should be in spectrum converter, for RGB factor will
    // be 0.33..3, but for other it will be 0.25
    Float avgAlbedo = albedo.Sum() * Float(0.3333);
    return Float(1) - MISRatio(avgAlbedo);
}

template <class SC>
MR_GF_DEF
RayConeSurface UnrealMaterial<SC>::RefractRayCone(const RayConeSurface& r,
                                                  const Vector3&) const
{
    return r;
}

template <class SC>
MR_GF_DEF
bool UnrealMaterial<SC>::IsAllTexturesAreResident(const Surface& surface,
                                                  const DataSoA& soa,
                                                  MaterialKey mk)
{
    const auto& albedoTex = soa.dAlbedo[mk.FetchIndexPortion()];
    const auto& roughnessTex = soa.dRoughness[mk.FetchIndexPortion()];
    const auto& specularTex = soa.dSpecular[mk.FetchIndexPortion()];
    const auto& metallicTex = soa.dMetallic[mk.FetchIndexPortion()];
    const auto& normalMapTex = soa.dNormalMaps[mk.FetchIndexPortion()];

    bool allResident = true;
    allResident &= roughnessTex.IsResident(surface.uv, surface.dpdx, surface.dpdy);
    allResident &= metallicTex.IsResident(surface.uv, surface.dpdx, surface.dpdy);
    allResident &= specularTex.IsResident(surface.uv, surface.dpdx, surface.dpdy);
    allResident &= albedoTex.IsResident(surface.uv, surface.dpdx, surface.dpdy);
    if(normalMapTex)
    {
        allResident &= normalMapTex->IsResident(surface.uv, surface.dpdx, surface.dpdy);
    }
    return allResident;
}

}

namespace PassthroughMatDetail
{

template <class SC>
MR_PF_DEF
NormalMap PassthroughMaterial<SC>::GetNormalMap(const DataSoA&, MaterialKey)
{
    return std::nullopt;
}

template <class SC>
MR_PF_DEF_V
PassthroughMaterial<SC>::PassthroughMaterial(const SpectrumConverter&,
                                             const Surface& surface,
                                             const DataSoA&, MaterialKey)
    : surface(surface)
{}

template <class SC>
MR_PF_DEF
BxDFSample
PassthroughMaterial<SC>::SampleBxDF(const Vector3& wO,
                                    RNGDispenser&) const
{
    return BxDFSample
    {
        .wI     = Ray(-wO, surface.position),
        .pdf    = Float(1.0),
        .eval   = BxDFEval
        {
            .reflectance     = Spectrum(1.0),
            .isPassedThrough = true,
            .isDispersed     = false
        }
    };
}

template <class SC>
MR_PF_DEF
Float
PassthroughMaterial<SC>::Pdf(const Ray&, const Vector3&) const
{
    // We can not sample this
    return Float(0);
}

template <class SC>
MR_PF_DEF
BxDFEval PassthroughMaterial<SC>::Evaluate(const Ray&, const Vector3&) const
{
    return BxDFEval
    {
        .reflectance     = Spectrum(1),
        .isPassedThrough = true,
        .isDispersed     = false
    };
}

template <class SC>
MR_PF_DEF
bool PassthroughMaterial<SC>::IsEmissive() const
{
    return false;
}

template <class SC>
MR_PF_DEF
Spectrum PassthroughMaterial<SC>::Emit(const Vector3&) const
{
    return Spectrum::Zero();
}

template <class SC>
MR_PF_DEF
Float PassthroughMaterial<SC>::Specularity() const
{
    return Float(1);
}

template <class SC>
MR_PF_DEF
RayConeSurface PassthroughMaterial<SC>::RefractRayCone(const RayConeSurface& r,
                                                       const Vector3&) const
{
    return r;
}

template <class SC>
MR_PF_DEF
bool PassthroughMaterial<SC>::IsAllTexturesAreResident(const Surface&, const DataSoA&,
                                                       MaterialKey)
{
    return true;
}

}
