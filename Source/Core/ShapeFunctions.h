#pragma once

#include "Vector.h"
#include "Types.h"

namespace ShapeFunctions
{
namespace Triangle
{
    static constexpr uint32_t TRI_VERTEX_COUNT = 3;

    MRAY_HYBRID AABB3   BoundingBox(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Float   Area(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Vector3 Normal(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Vector3 CalculateTangent(const Vector3& p0, const Vector3& p1, const Vector3& p2,
                                         const Vector2& uv0, const Vector2& uv1, const Vector2& uv2);
}
}

namespace ShapeFunctions
{

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 Triangle::BoundingBox(Span<const Vector3, TRI_VERTEX_COUNT> positions)
{
    AABB3 aabb(positions[0], positions[0]);
    aabb.SetMin(Vector3::Min(aabb.Min(), positions[1]));
    aabb.SetMin(Vector3::Min(aabb.Min(), positions[2]));
    aabb.SetMax(Vector3::Max(aabb.Max(), positions[1]));
    aabb.SetMax(Vector3::Max(aabb.Max(), positions[2]));
    return aabb;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float Triangle::Area(Span<const Vector3, TRI_VERTEX_COUNT> positions)
{
    Vector3 e0 = positions[1] - positions[0];
    Vector3 e1 = positions[2] - positions[0];

    return Vector3::Cross(e0, e1).Length() * static_cast<Float>(0.5);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Triangle::Normal(Span<const Vector3, TRI_VERTEX_COUNT> positions)
{
    Vector3 e0 = positions[1] - positions[0];
    Vector3 e1 = positions[2] - positions[0];

    return Vector3::Cross(e0, e1).Normalize();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Triangle::CalculateTangent(const Vector3& p0, const Vector3& p1, const Vector3& p2,
                                   const Vector2& uv0, const Vector2& uv1, const Vector2& uv2)
{
    // Edges (Tri is CCW)
    Vector3 vec0 = p1 - p0;
    Vector3 vec1 = p2 - p0;

    Vector2 dUV0 = uv1 - uv0;
    Vector2 dUV1 = uv2 - uv0;

    Float t = (dUV0[0] * dUV1[1] -
               dUV1[0] * dUV0[1]);
    // UVs are not set or bad just return NaN
    if(t == 0.0f) return Vector3(std::numeric_limits<Float>::quiet_NaN());
    // Calculate as normal
    float r = 1.0f / t;
    Vector3 tangent = r * (dUV1[1] * vec0 - dUV0[1] * vec1);
    // Check if the tangent, bi-tangent determine
    // a right handed coordinate system
    return (t < 0) ? -tangent : tangent;
}

}