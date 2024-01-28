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

}