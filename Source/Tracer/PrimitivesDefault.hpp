#pragma once

#pragma once

namespace DefaultSphereDetail
{

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Sphere<T>::Sphere(const T& transform,
                  const SphereData& data,
                  PrimitiveKey key)
    : center(data.centers[key.FetchIndexPortion()])
    , radius(data.radius[key.FetchIndexPortion()])
    , transformContext(transform)
{}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<SphereIntersection> Sphere<T>::Intersects(const Ray& ray, bool cullBackFace) const
{
    Ray transformedRay = transformContext.get().InvApply(ray);

    // Intersection
    Float t;
    Vector3 hitPos;
    bool intersects = transformedRay.IntersectsSphere(hitPos, t,
                                                      center, radius);
    if(!intersects) return std::nullopt;

    Vector3 unitDir = (hitPos - center).Normalize();
    Vector2 hit = Graphics::CartesianToUnitSpherical(unitDir);

    bool isInside = (transformedRay.Pos() - center).Length() < radius;
    if(cullBackFace && isInside)
        return std::nullopt;

    return SphereIntersection
    {
        .hit = hit,
        .t = t
    };
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BasicSurface> Sphere<T>::SampleSurface(RNGDispenser& rng) const
{
    // Sampling sphere surface
    // http://mathworld.wolfram.com/SpherePointPicking.html
    Vector2 xi = rng.NextFloat2D<0>();

    Float theta = Float(2) * MathConstants::Pi<Float>() * xi[0];
    Float cosPhi = Float(2) * xi[1] - Float(1);
    Float sinPhi = sqrtf(fmaxf(Float(0), Float(1) - cosPhi * cosPhi));

    auto sinCosTheta = Vector2(std::sin(theta), std::cos(theta));
    auto sinCosPhi = Vector2(sinPhi, cosPhi);
    Vector3 unitPos = Graphics::UnitSphericalToCartesian(sinCosTheta,
                                                         sinCosPhi);

    // Calculate PDF
    // Approximate the area with the determinant
    Float pdf = Float(1) / GetSurfaceArea();

    Vector3 sphrLoc = center + radius * unitPos;
    sphrLoc = transformContext.get().ApplyP(sphrLoc);
    Vector3 normal = transformContext.get().ApplyN(unitPos.Normalize());

    return SampleT<BasicSurface>
    {
        .value = BasicSurface
        {
            .position = sphrLoc,
            .normal = normal
        },
        .pdf = pdf
    };
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Float Sphere<T>::PdfSurface(const Hit&) const
{
    return Float(1) / GetSurfaceArea();
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Float Sphere<T>::GetSurfaceArea() const
{
    // https://math.stackexchange.com/questions/942561/surface-area-of-transformed-sphere
    static constexpr Float p = Float(8) / Float(5);
    static constexpr Float pRecip = Float(1) / p;

    Vector3 semiAxes = radius * transformContext.get().Scale();
    Float approxArea = std::pow(semiAxes[1] * semiAxes[2], p);
    approxArea += std::pow(semiAxes[2] * semiAxes[0], p);
    approxArea += std::pow(semiAxes[0] * semiAxes[1], p);
    approxArea *= Float(0.33333333);
    approxArea = std::pow(approxArea, pRecip);
    approxArea *= Float(4) * MathConstants::Pi<Float>();
    return approxArea;
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 Sphere<T>::GetAABB() const
{
    AABB3 aabb = Shape::Sphere::BoundingBox(center, radius);
    return transformContext.get().Apply(aabb);
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Sphere<T>::GetCenter() const
{
    return transformContext.get().ApplyP(center);
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t Sphere<T>::Voxelize(Span<uint64_t>&,
                             Span<Vector2us>&,
                             bool,
                             const VoxelizationParameters&) const
{
    // TODO: Implement voxelization
    return 0;
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<BasicSurface> Sphere<T>::SurfaceFromHit(const Hit& hit) const
{
    // Convert spherical hit to cartesian
    Vector3 normal = Graphics::UnitSphericalToCartesian(hit);
    // Calculate local position using the normal
    // then convert it to world position
    Vector3 position = center + normal * radius;
    position = transformContext.get().ApplyP(position);
    // Generate Geometric world space normal
    // In sphere case it is equivalent to the normal
    Vector3 geoNormal = transformContext.get().ApplyN(normal).Normalize();
    return BasicSurface
    {
        .position = position,
        .normal = geoNormal
    };
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<SphereHit> Sphere<T>::ProjectedHit(const Vector3& point) const
{
    using namespace Shape::Triangle;
    Vector3 projPoint = (point - center).Normalize();
    Vector2 sphrCoords = Graphics::CartesianToUnitSpherical(projPoint);

    return Hit(sphrCoords);
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 Sphere<T>::SurfaceParametrization(const Hit& hit) const
{
    // Gen UV
    Vector2 uv = hit;
    // theta is [0, 2 * pi], normalize
    uv[0] *= Float(0.5) * MathConstants::InvPi<Float>();
    // phi is [-pi/2, pi/2], normalize
    uv[1] = uv[1] * MathConstants::InvPi<Float>() + Float(0.5);
    return uv;
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
const T& Sphere<T>::GetTransformContext() const
{
    return transformContext;
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Sphere<T>::GenerateSurface(EmptySurface&,
                                // Inputs
                                const Hit&,
                                const Ray&,
                                const RayDiff&) const
{}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Sphere<T>::GenerateSurface(BasicSurface& result,
                                // Inputs
                                const Hit& hit,
                                const Ray&,
                                const RayDiff&) const
{
    result = *SurfaceFromHit(hit);
}

template<TransformContextC T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Sphere<T>::GenerateSurface(DefaultSurface& result,
                                // Inputs
                                const Hit& hit,
                                const Ray& ray,
                                const RayDiff&) const
{
    // Convert spherical hit to cartesian
    Vector3 normal = Graphics::UnitSphericalToCartesian(hit);
    Vector3 geoNormal = transformContext.get().ApplyN(normal).Normalize();
    // Calculate local position using the normal
    // then convert it to world position
    Vector3 position = center + normal * radius;
    position = transformContext.get().ApplyP(position);

    // Align this normal to Z axis to define tangent space rotation
    Quaternion tbn = Quaternion::RotationBetweenZAxis(normal.Normalize()).Conjugate();

    // Spheres are always two sided, check if we are inside
    bool backSide = (geoNormal.Dot(ray.Dir()) > Float(0));
    if(backSide)
    {
        geoNormal = -geoNormal;
        // Change the tbn rotation so that Z is on opposite direction
        // TODO: here flipping Z would change the handedness of the
        // coordinate system
        // Just adding the 180degree rotation with the tangent axis
        // to the end which should be fine I guess?
        static constexpr auto TANGENT_ROT = Quaternion(0, 1, 0, 0);
        tbn = TANGENT_ROT * tbn;
    }
    Vector2 uv = SurfaceParametrization(hit);

    result = DefaultSurface
    {
        .position = position,
        .geoNormal = geoNormal,
        .shadingTBN = tbn,
        .uv = uv,
        .dpdu = Vector2::Zero(),
        .dpdv = Vector2::Zero(),
        .backSide = backSide
    };
}

}