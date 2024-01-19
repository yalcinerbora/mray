#pragma once

namespace LightDetail
{

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
PrimLight<P, SC>::PrimLight(const typename SC::Converter& specTransformer,
                            const P& p, const LightData& soa, LightId id)
    : prim(p)
    , radiance(specTransform, soa.dRadiances[id.FetchIndexPortion()])
    , initialMedium(soa.dMediumIds[id.FetchIndexPortion()])
    , isTwoSided(soa.dIsTwoSidedFlags[id.FetchIndexPortion()])
{}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> PrimLight<P, SC>::SampleProjSurface(RNGDispenser& rng,
                                                     const Vector3& dir) const
{
    SampleT<BasicSurface> surfaceSample = p.SampleSurface(rng);

    Float NdL = surfaceSample.sampledResult.geoNormal.Dot(-dir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    Float pdf = (NdL == 0) ? Float{0.0} : surfaceSample.pdf / NdL;
    pdf *= dir.LengthSqr();

    return SampleT<Vector3>
    {
        .sampledResult = surfaceSample.sampledResult.position,
        .pdf = pdf
    };
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float PrimLight<P, SC>::PdfProjSurface(const Vector3& position,
                                       const Vector3& dir) const
{
    // Project point to surface (function assumes
    Optional<Vector3> normal = p.ProjectedNormal(position);
    if(!normal) return Float{0};

    Float area = p.PdfSurface(position);
    Float NdL = projectedNormal.Dot(-dir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    pdf = (NdL == 0) ? Float{0.0} : pdf / (NdL * area);
    pdf *= dir.LengthSqr();
    return pdf;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PrimLight<P, SC>::SampleProjSurfaceRNCount() const
{
    return p.SampleRNCount();
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> PrimLight<P, SC>::SampleRay(RNGDispenser& dispenser,
                                          ) const
{
    // What is the probability?
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float PrimLight<P, SC>::PdfRay(const Ray&) const
{

}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PrimLight<P, SC>::SampleRayRNCount() const
{
    4
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum PrimLight<P, SC>::Emit(const Vector3& wO,
                                const typename P::Hit& hitParams) const
{
    // Find
    Vector2 uv = radiance.Constant()
                    ? Vector2::Zero()
                    : p.ProjectedHit(hitParams);
    return radiance(uv);
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum PrimLight<P, SC>::Emit(const Vector3& wO,
                                const Vector3& surfacePoint) const
{
    using Hit = typename P::Hit;
    Optional<Hit> hit = p.ProjectedSurfaceParametrization(surfacePoint);
    if(!hit) return Spectrum::Zero();

    Vector2 uv = radiance.Constant()
                    ? Vector2::Zero()
                    : hit.value();
    return radiance(uv);
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
bool PrimLight<P, SC>::IsPrimitiveBackedLight() const
{
    return true;
}

}