#pragma once

#include "Vector.h"
#include "Types.h"

#include "Core/GraphicsFunctions.h"

namespace Shape
{
namespace Triangle
{
    static constexpr uint32_t TRI_VERTEX_COUNT = 3;

    MRAY_HYBRID AABB3   BoundingBox(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Float   Area(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Vector3 Normal(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MRAY_HYBRID Vector3 CalculateTangent(const Vector3& p0Normal,
                                         const std::array<Vector3, 3>& positions,
                                         const std::array<Vector2, 3>& uvs);
    MRAY_HYBRID Vector3 Project(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                const Vector3& point);
    MRAY_HYBRID Vector3 PointToBarycentrics(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                            const Vector3& point);
}

namespace Sphere
{
    MRAY_HYBRID AABB3 BoundingBox(const Vector3& center, Float radius);
}

namespace Polygon
{
    // Given a polygon with vertices "vertices"
    // calculate triangulation of such polygon via "ear clipping"
    // method. Polygon winding order must be clockwise
    template<size_t N>
    MRAY_HYBRID
    constexpr void ClipEars(Span<Vector3ui, N - 2> localIndicesOut,
                            Span<const Vector3, N> vertices);
}
}

namespace Shape
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
Vector3 Triangle::CalculateTangent(const Vector3& p0Normal,
                                   const std::array<Vector3, 3>& p,
                                   const std::array<Vector2, 3>& uv)
{
    // Edges (Tri is CCW)
    Vector3 e0 = p[1] - p[0];
    Vector3 e1 = p[2] - p[0];

    Vector2 dUV0 = uv[1] - uv[0];
    Vector2 dUV1 = uv[2] - uv[0];

    Float det = (dUV0[0] * dUV1[1] -
                 dUV1[0] * dUV0[1]);
    // Numeric precision issues
    if(std::abs(det) < MathConstants::Epsilon<Float>())
    {
        // From PBRT-v4
        // https://github.com/mmp/pbrt-v4/blob/779d1a78b74aab393853544198189729434121b5/src/pbrt/shapes.h#L911
        // Basically just generate random tangent related to shading normal
        Vector3 normal = Shape::Triangle::Normal(p);
        // Triangle is degenerate (line probably)
        normal = (normal.HasNaN()) ? p0Normal : normal;
        // If tangent is still has issues, we tried...
        // return it
        Vector3 tangent = Vector3::OrthogonalVector(normal);
        if(tangent.HasNaN())
            __debugbreak();
        return tangent;
    }
    if(det == 0)
        __debugbreak();
    // Calculate as normal
    Float r = Float(1) / det;
    Vector3 tangent = r * (dUV1[1] * e0 - dUV0[1] * e1);
    // Check if the tangent, bi-tangent determine
    // a right handed coordinate system
    //return tangent;
    return (det < Float(0)) ? -tangent : tangent;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Triangle::Project(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                          const Vector3& point)
{
    Vector3 n = Normal(positions);
    Vector3 dir = point - positions[0];
    n = Graphics::Orient(n, dir);
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
    Float denom = Float(1) / (d00 * d11 - d01 * d01);
    Float a = (d11 * d20 - d01 * d21) * denom;
    Float b = (d00 * d21 - d01 * d20) * denom;
    Float c = Float(1) - a - b;
    return Vector3(a, b, c);
}

MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 Sphere::BoundingBox(const Vector3& center, Float radius)
{
    return AABB3(center - radius, center + radius);
}

template<size_t N>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Polygon::ClipEars(Span<Vector3ui, N - 2> localIndicesOut,
                                 Span<const Vector3, N> vertices)
{
    // We "expand" the vertices to previous/next we will access elements
    // via this
    auto indices = StaticVector<uint32_t, N>(StaticVecSize(N));
    std::iota(indices.begin(), indices.end(), 0u);
    //
    const auto Next = [&indices](uint32_t i) -> uint32_t
    {
        return ((++i) >= indices.size()) ? 0u : i;
    };
    const auto Prev = [&indices](uint32_t i) -> uint32_t
    {
        return (i == 0u) ? uint32_t(indices.size() - 1u) : i - 1u;
    };
    //
    using MathConstants::Epsilon;
    const auto IsConvexEdge = [&](Vector3ui i) -> bool
    {
        Vector3 e0 = vertices[i[2]] - vertices[i[1]];
        Vector3 e1 = vertices[i[0]] - vertices[i[1]];
        return Vector3::Cross(e0, e1).LengthSqr() >= -Epsilon<Float>();
    };
    // Basic ear clipping algorithm
    // Traverse the contigious triplets
    uint32_t writeIndex = 0;
    for(uint32_t i = 1; indices.size() > 2; i = Next(i))
    {
        Vector3ui triplet(indices[Prev(i)],
                          indices[i],
                          indices[Next(i)]);
        if(IsConvexEdge(triplet))
        {
            // Write the triplet
            localIndicesOut[writeIndex++] = triplet;
            // Now collapse the array
            indices.remove(&indices[i]);
            // When we continue, do not decrement "i"
            // to compansate the Next(..) function
            // Given a equilateral polygon, this will
            // give similar results to Delunay (sometimes).
            // Otherwise, it will generate fan triangulation
            // which is arguably worse?
        }
    }
}
}