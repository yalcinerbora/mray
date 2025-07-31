#pragma once

#include "Vector.h"
#include "Types.h"

#include "Core/GraphicsFunctions.h"

namespace Shape
{
namespace Triangle
{
    static constexpr uint32_t TRI_VERTEX_COUNT = 3;

    MR_HF_DECL AABB3    BoundingBox(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MR_HF_DECL Float    Area(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MR_HF_DECL Vector3  Normal(Span<const Vector3, TRI_VERTEX_COUNT> positions);
    MR_HF_DECL Vector3  CalculateTangent(const Vector3& p0Normal,
                                         const std::array<Vector3, 3>& positions,
                                         const std::array<Vector2, 3>& uvs);
    MR_HF_DECL Vector3  Project(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                const Vector3& point);
    MR_HF_DECL Vector3  PointToBarycentrics(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                            const Vector3& point);
}

namespace Sphere
{
    MR_HF_DECL AABB3 BoundingBox(const Vector3& center, Float radius);
}

namespace Polygon
{
    // Given a polygon with vertices "vertices"
    // calculate triangulation of such polygon via "ear clipping"
    // method. Polygon winding order must be clockwise
    template<size_t N>
    MR_HF_DECL constexpr
    void ClipEars(Span<Vector3ui, N - 2> localIndicesOut,
                  Span<const Vector3, N> vertices,
                  const Vector3& normal);
}
}

namespace Shape
{

MR_HF_DEF
AABB3 Triangle::BoundingBox(Span<const Vector3, TRI_VERTEX_COUNT> positions)
{
    AABB3 aabb(positions[0], positions[0]);
    aabb.SetMin(Math::Min(aabb.Min(), positions[1]));
    aabb.SetMin(Math::Min(aabb.Min(), positions[2]));
    aabb.SetMax(Math::Max(aabb.Max(), positions[1]));
    aabb.SetMax(Math::Max(aabb.Max(), positions[2]));
    return aabb;
}

MR_HF_DEF
Float Triangle::Area(Span<const Vector3, TRI_VERTEX_COUNT> positions)
{
    Vector3 e0 = positions[1] - positions[0];
    Vector3 e1 = positions[2] - positions[0];
    return Math::Length(Math::Cross(e0, e1)) * static_cast<Float>(0.5);
}

MR_HF_DEF
Vector3 Triangle::Normal(Span<const Vector3, TRI_VERTEX_COUNT> positions)
{
    Vector3 e0 = positions[1] - positions[0];
    Vector3 e1 = positions[2] - positions[0];
    return Math::Normalize(Math::Cross(e0, e1));
}

MR_HF_DEF
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
    if(Math::Abs(det) < MathConstants::Epsilon<Float>())
    {
        // From PBRT-v4
        // https://github.com/mmp/pbrt-v4/blob/779d1a78b74aab393853544198189729434121b5/src/pbrt/shapes.h#L911
        // Basically just generate random tangent related to shading normal
        Vector3 normal = Shape::Triangle::Normal(p);
        // Triangle is degenerate (line probably)
        normal = (Math::IsFinite(normal)) ? normal : p0Normal;
        // If tangent is still has issues, we tried...
        // return it
        Vector3 tangent = Graphics::OrthogonalVector(normal);
        return tangent;
    }
    // Calculate as normal
    Float r = Float(1) / det;
    Vector3 tangent = r * (dUV1[1] * e0 - dUV0[1] * e1);
    // Check if the tangent, bi-tangent determine
    // a right handed coordinate system
    //return tangent;
    return (det < Float(0)) ? -tangent : tangent;
}

MR_HF_DEF
Vector3 Triangle::Project(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                          const Vector3& point)
{
    Vector3 n = Normal(positions);
    Vector3 dir = point - positions[0];
    n = Graphics::Orient(n, dir);
    return point - Math::Dot(dir, n) * n;
}

MR_HF_DEF
Vector3 Triangle::PointToBarycentrics(Span<const Vector3, TRI_VERTEX_COUNT> positions,
                                      const Vector3& point)
{
    // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    Vector3 e0 = positions[1] - positions[0];
    Vector3 e1 = positions[2] - positions[0];
    Vector3 v = point - positions[0];

    Float d00 = Math::Dot(e0, e0);
    Float d01 = Math::Dot(e0, e1);
    Float d11 = Math::Dot(e1, e1);
    Float d20 = Math::Dot(v, e0);
    Float d21 = Math::Dot(v, e1);
    Float denom = Float(1) / (d00 * d11 - d01 * d01);
    Float a = (d11 * d20 - d01 * d21) * denom;
    Float b = (d00 * d21 - d01 * d20) * denom;
    Float c = Float(1) - a - b;
    return Vector3(a, b, c);
}

MR_HF_DEF
AABB3 Sphere::BoundingBox(const Vector3& center, Float radius)
{
    return AABB3(center - radius, center + radius);
}

template<size_t N>
MR_HF_DEF constexpr
void Polygon::ClipEars(Span<Vector3ui, N - 2> localIndicesOut,
                       Span<const Vector3, N> vertices,
                       const Vector3& normal)
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
    using namespace MathConstants;
    const auto IsConvexEdge = [&](Vector3ui i) -> bool
    {
        Vector3 e0 = vertices[i[2]] - vertices[i[1]];
        Vector3 e1 = vertices[i[0]] - vertices[i[1]];
        Vector3 nV = Math::Cross(e0, e1);
        return Math::Dot(nV, normal) >= Float(0);
    };
    // Basic ear clipping algorithm
    // Traverse the contigious triplets
    uint32_t iter = 0;
    uint32_t writeIndex = 0;
    for(uint32_t i = 1; indices.size() > 2; i = Next(i))
    {
        // Maybe triangles are degenerate do a 2 * N pass,
        // after that start accepting triangles
        bool degenerateTri = (iter++) >= (2 * N);
        Vector3ui triplet(indices[Prev(i)],
                          indices[i],
                          indices[Next(i)]);
        if(IsConvexEdge(triplet) || degenerateTri)
        {
            // Write the triplet
            localIndicesOut[writeIndex++] = triplet;
            // Now collapse the array
            indices.remove(&indices[i]);
            // When we continue, do not decrement "i"
            // to compensate the Next(..) function
            // Given a equilateral polygon, this will
            // give similar results to Delunay (sometimes).
            // Otherwise, it will generate fan triangulation
            // which is arguably worse?
        }
    }
}
}