#pragma once

namespace LightDetail
{

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
PrimLight<P, SC>::PrimLight(const typename SC::Converter& specTransformer,
                            const P& p, const LightData& soa, LightId id)
    : prim(p)
    , radiance(specTransformer, soa.dRadiances[id.FetchIndexPortion()])
    , initialMedium(soa.dMediumIds[id.FetchIndexPortion()])
    , isTwoSided(soa.dIsTwoSidedFlags[id.FetchIndexPortion()])
{}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> PrimLight<P, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                    const Vector3& distantPoint,
                                                    const Vector3& dir) const
{
    SampleT<BasicSurface> surfaceSample = prim.SampleSurface(rng);

    Float NdL = surfaceSample.sampledResult.geoNormal.Dot(-dir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    Float pdf = (NdL == 0) ? Float{0.0} : surfaceSample.pdf / NdL;
    pdf *= (distantPoint - surfaceSample.sampledresult.position).LengthSqr();

    return SampleT<Vector3>
    {
        .sampledResult = surfaceSample.sampledResult.position,
        .pdf = pdf
    };
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float PrimLight<P, SC>::PdfSolidAngle(const typename P::Hit& hit,
                                      const Vector3& distantPoint,
                                      const Vector3& dir) const
{
    // Project point to surface (function assumes
    Optional<BasicSurface> surface = prim.SurfaceFromHit(hit);
    if(!surface) return Float{0};

    Float pdf = prim.PdfSurface(hit);
    Float NdL = surface.normal.Dot(-dir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    pdf = (NdL == 0) ? Float{0.0} : pdf / NdL;
    pdf *= (distantPoint - surface.position);
    return pdf;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PrimLight<P, SC>::SampleSolidAngleRNCount() const
{
    return prim.SampleRNCount();
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> PrimLight<P, SC>::SampleRay(RNGDispenser& dispenser) const
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
    return 4;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum PrimLight<P, SC>::Emit(const Vector3& wO,
                                const typename P::Hit& hitParams) const
{
    // Find
    Vector2 uv = radiance.Constant()
                    ? Vector2::Zero()
                    : prim.ProjectedHit(hitParams);
    return radiance(uv);
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum PrimLight<P, SC>::Emit(const Vector3& wO,
                                const Vector3& surfacePoint) const
{
    using Hit = typename P::Hit;
    Optional<Hit> hit = prim.ProjectedHit(surfacePoint);
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