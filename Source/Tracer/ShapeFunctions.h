#pragma once

#include "Core/Vector.h"
#include "Core/Types.h"

namespace ShapeFunctions
{
namespace Triangle
{
    static constexpr PrimitiveId TRI_VERTEX_COUNT = 3;

    MRAY_HYBRID AABB3   BoundingBox(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Float   Area(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Vector3 Normal(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Vector3 Project(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                const Vector3& point);
    MRAY_HYBRID Vector3 PointToBarycentrics(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                            const Vector3& point);
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
Vector3 Triangle::Project(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                          const Vector3& point)
{
    Vector3 n = Normal(positions);
    Vector3 dir = point - positions[0];
    n = GraphicsFunctions::Orient(n, dir);
    return point - dir.Dot(n) * n;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Triangle::PointToBarycentrics(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                      const Vector3& point)
{
    // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    Vector3 e0 = positions[1] - positions[0];
    Vector3 e1 = positions[2] - positions[0];
    Vector3 v = point - positions[0];

    Float d00 = e0.Dot(e0);
    Float d01 = e0.Dot(e1);
    Float d11 = e1.Dot(e1);
    Float d20 = v.Dot(e0);
    Float d21 = v.Dot(e1);
    Float denom = 1.0f / (d00 * d11 - d01 * d01);
    Float a = (d11 * d20 - d01 * d21) * denom;
    Float b = (d00 * d21 - d01 * d20) * denom;
    Float c = 1.0f - a - b;
    return Vector3(a, b, c);
}