#pragma once

#include "PrimitivesDefault.h"
#include "Core/ShapeFunctions.h"

namespace DefaultSphereDetail
{

template<TransformContextC T>
MR_HF_DEF
Sphere<T>::Sphere(const T& transform,
                  const SphereData& data,
                  PrimitiveKey key)
    : center(data.centers[key.FetchIndexPortion()])
    , radius(data.radius[key.FetchIndexPortion()])
    , transformContext(transform)
{}

template<TransformContextC T>
MR_HF_DEF
Optional<SphereIntersection> Sphere<T>::Intersects(const Ray& ray, bool cullBackFace) const
{
    Ray transformedRay = transformContext.get().InvApply(ray);

    // Intersection
    Float t;
    Vector3 hitPos;
    bool intersects = transformedRay.IntersectsSphere(hitPos, t,
                                                      center, radius);
    if(!intersects) return std::nullopt;

    Vector3 unitDir = Math::Normalize(hitPos - center);
    Vector2 hit = Graphics::CartesianToUnitSpherical(unitDir);

    bool isInside = Math::Length(transformedRay.pos - center) < radius;
    if(cullBackFace && isInside)
        return std::nullopt;

    return SphereIntersection
    {
        .hit = hit,
        .t = t
    };
}

template<TransformContextC T>
MR_HF_DEF
SampleT<BasicSurface> Sphere<T>::SampleSurface(RNGDispenser& rng) const
{
    // Sampling sphere surface
    // http://mathworld.wolfram.com/SpherePointPicking.html
    Vector2 xi = rng.NextFloat2D<0>();

    Float theta = Float(2) * MathConstants::Pi<Float>() * xi[0];
    Float cosPhi = Float(2) * xi[1] - Float(1);
    Float sinPhi = sqrtf(fmaxf(Float(0), Float(1) - cosPhi * cosPhi));

    auto sinCosTheta = Vector2(Math::SinCos(theta));
    auto sinCosPhi = Vector2(sinPhi, cosPhi);
    Vector3 unitPos = Graphics::UnitSphericalToCartesian(sinCosTheta,
                                                         sinCosPhi);

    // Calculate PDF
    // Approximate the area with the determinant
    Float pdf = Float(1) / GetSurfaceArea();

    Vector3 sphrLoc = center + radius * unitPos;
    sphrLoc = transformContext.get().ApplyP(sphrLoc);
    Vector3 normal = transformContext.get().ApplyN(Math::Normalize(unitPos));

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
MR_HF_DEF
Float Sphere<T>::PdfSurface(const Hit&) const
{
    return Float(1) / GetSurfaceArea();
}

template<TransformContextC T>
MR_HF_DEF
Float Sphere<T>::GetSurfaceArea() const
{
    // https://math.stackexchange.com/questions/942561/surface-area-of-transformed-sphere
    static constexpr Float p = Float(8) / Float(5);
    static constexpr Float pRecip = Float(1) / p;

    Vector3 semiAxes = radius * transformContext.get().Scale();
    Float approxArea = Math::Pow(semiAxes[1] * semiAxes[2], p);
    approxArea += Math::Pow(semiAxes[2] * semiAxes[0], p);
    approxArea += Math::Pow(semiAxes[0] * semiAxes[1], p);
    approxArea *= Float(0.33333333);
    approxArea = Math::Pow(approxArea, pRecip);
    approxArea *= Float(4) * MathConstants::Pi<Float>();
    return approxArea;
}

template<TransformContextC T>
MR_HF_DEF
AABB3 Sphere<T>::GetAABB() const
{
    AABB3 aabb = Shape::Sphere::BoundingBox(center, radius);
    return transformContext.get().Apply(aabb);
}

template<TransformContextC T>
MR_HF_DEF
Vector3 Sphere<T>::GetCenter() const
{
    return transformContext.get().ApplyP(center);
}

template<TransformContextC T>
MR_HF_DEF
uint32_t Sphere<T>::Voxelize(Span<uint64_t>&,
                             Span<Vector2us>&,
                             bool,
                             const VoxelizationParameters&) const
{
    // TODO: Implement voxelization
    return 0;
}

template<TransformContextC T>
MR_HF_DEF
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
    Vector3 geoNormal = Math::Normalize(transformContext.get().ApplyN(normal));
    return BasicSurface
    {
        .position = position,
        .normal = geoNormal
    };
}

template<TransformContextC T>
MR_HF_DEF
Optional<SphereHit> Sphere<T>::ProjectedHit(const Vector3& point) const
{
    using namespace Shape::Triangle;
    Vector3 projPoint = Math::Normalize(point - center);
    Vector2 sphrCoords = Graphics::CartesianToUnitSpherical(projPoint);

    return Hit(sphrCoords);
}

template<TransformContextC T>
MR_HF_DEF
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
MR_HF_DEF
const T& Sphere<T>::GetTransformContext() const
{
    return transformContext;
}

template<TransformContextC T>
MR_HF_DEF
void Sphere<T>::GenerateSurface(EmptySurface&,
                                RayConeSurface& rayConeSurface,
                                // Inputs
                                const NormalMap&,
                                const Hit&,
                                const Ray&,
                                const RayCone& rayCone) const
{
    rayConeSurface = RayConeSurface
    {
        .rayConeFront   = rayCone,
        .rayConeBack    = rayCone,
        .betaN          = 0
    };
}

template<TransformContextC T>
MR_HF_DEF
void Sphere<T>::GenerateSurface(BasicSurface& result,
                                RayConeSurface& rayConeSurface,
                                // Inputs
                                const NormalMap&,
                                const Hit& hit,
                                const Ray&,
                                const RayCone& rayCone) const
{
    result = *SurfaceFromHit(hit);
    rayConeSurface = RayConeSurface
    {
        .rayConeFront   = rayCone,
        .rayConeBack    = rayCone,
        .betaN          = 0
    };
}

template<TransformContextC T>
MR_GF_DEF
void Sphere<T>::GenerateSurface(DefaultSurface& result,
                                RayConeSurface& rayConeSurface,
                                // Inputs
                                const NormalMap& normalMap,
                                const Hit& hit,
                                const Ray& ray,
                                const RayCone& rayCone) const
{
    const auto& transform = transformContext.get();

    // Convert spherical hit to cartesian
    Vector3 normal = Graphics::UnitSphericalToCartesian(hit);
    Vector3 geoNormal = Math::Normalize(transform.ApplyN(normal));
    // Calculate local position using the normal
    // then convert it to world position
    Vector3 position = center + normal * radius;
    position = transform.ApplyP(position);

    // Align this normal to Z axis to define tangent space rotation
    Quaternion tbn = Quaternion::RotationBetweenZAxis(Math::Normalize(normal)).Conjugate();

    // Spheres are always two sided, check if we are inside
    Vector3 rDir = Math::Normalize(ray.dir);
    Float dDotN = Math::Dot(geoNormal, rDir);
    bool backSide = (dDotN > Float(0));
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


    // Curvature / texture differentials
    // https://www.jcgt.org/published/0010/01/01/
    // Sphere implementation is not there
    // Equation 5
    Vector3 centerWorld = transform.ApplyP(center);
    // The sphere may be transformed via non-uniform scale
    // so we need to calculate "radius" from the world positions
    Float r = Math::Length(centerWorld - position);
    Float betaN = Float(-1) * Math::Abs(rayCone.width) / (dDotN * r);
    betaN = backSide ? -betaN : betaN;
    // Texture space differentials
    auto [a1, a2] = rayCone.Project(geoNormal, rDir);
    //
    auto TexGradient = [&](Vector3 offset)
    {
        // Sphere may have been transformed and skewed
        // So we need to calculate UVs in local space
        // But this is costly.
        // TODO: Optimize this maybe?
        Vector3 offsetP = (position + offset);
        Vector3 n = Math::Normalize(transform.InvApplyP(offsetP) - center);
        Vector2 sphrCoords = Graphics::CartesianToUnitSpherical(n);
        Vector2 texCoord = SurfaceParametrization(sphrCoords);
        return texCoord - uv;
    };
    Vector2 dpdx = TexGradient(a1);
    Vector2 dpdy = TexGradient(a2);

    if(normalMap)
    {
        Vector3 n = Math::Normalize((*normalMap)(uv, dpdx, dpdy));
        tbn = Quaternion::RotationBetweenZAxis(n).Conjugate() * tbn;
    }
    result = DefaultSurface
    {
        .position = position,
        .geoNormal = geoNormal,
        .shadingTBN = tbn,
        .uv = uv,
        .dpdx = dpdx,
        .dpdy = dpdy,
        .backSide = backSide
    };
    // TODO:
    rayConeSurface = RayConeSurface
    {
        .rayConeFront   = rayCone,
        .rayConeBack    = rayCone,
        .betaN          = betaN
    };
}

}