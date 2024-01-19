#pragma once

namespace DefaultTriangleDetail
{

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Triangle<T>::Triangle(const T& transform,
                      const TriangleData& data,
                      PrimitiveId id)
    : data(data)
    , id(id)
    , positions{}
    , transformContext(transform)
{
    static constexpr auto VPP = ShapeFunctions::Triangle::TRI_VERTEX_COUNT;

    Vector3ui index = data.indexList[id];
    positions[0] = data.positions[index[0]];
    positions[1] = data.positions[index[1]];
    positions[2] = data.positions[index[2]];
    // Apply the transform
    positions[0] = transform.ApplyP(positions[0]);
    positions[1] = transform.ApplyP(positions[1]);
    positions[2] = transform.ApplyP(positions[2]);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
TriIntersection Triangle<T>::Intersects(const Ray& ray) const
{
    uint32_t primSubBatchId = data.subBatchTable.FindIndex(id);
    const bool cullBackface = data.cullFace[primSubBatchId];

    // Intersection
    float t;
    Vector3 baryCoords;
    bool intersects = false;
    bool intersects = ray.IntersectsTriangle(baryCoords, t,
                                             positions,
                                             cullBackface);

    TriIntersection result = (!intersects) ? std::nullopt_t : IntersectionT<TriHit>
    {
        .t = t,
        .hit = Vector2(baryCoords)
    };
    return result;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<BasicSurface> Triangle<T>::SampleSurface(const RNGDispenser& rng) const
{
    Vector2 xi = rng.NextFloat2D<0>();
    Float r1 = sqrt(xi[0]);
    Float r2 = xi[1];
    // Generate Random Barycentrics
    // Osada 2002
    // http://graphics.stanford.edu/courses/cs468-08-fall/pdf/osada.pdf
    Float a = 1 - r1;
    Float b = (1 - r2) * r1;
    Float c = r1 * r2;

    Vector3 position = (positions[0] * a +
                        positions[1] * b +
                        positions[2] * c);

    Float area = GetSurfaceArea();
    Float pdf = 1.0f / area;

    Vector3ui index = data.indexList[id];
    Quaternion q0 = data.tbnRotations[index[0]];
    Quaternion q1 = data.tbnRotations[index[1]];
    Quaternion q2 = data.tbnRotations[index[2]];
    Quaternion tbn = Quaternion::BarySLerp(q0, q1, q2, a, b);
    Vector3 normal = tbn.Conjugate().ApplyRotation(Vector3::ZAxis());

    using ShapeFunctions::Triangle::Normal;
    return SampleT<BasicSurface>
    {
        .pdf = pdf,
        .sampledResult = BasicSurface
        {
            .position = position,
            .geoNormal = normal
        }
    };
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Float Triangle<T>::PdfSurface(const Hit& hit) const
{
    Vector3 baryCoords = Vector3(hit[0], hit[1], 1 - hit[0] - hit[1]);
    if(baryCoords[0] < 0 || baryCoords[0] > 1 ||
       baryCoords[1] < 0 || baryCoords[1] > 1 ||
       baryCoords[2] < 0 || baryCoords[2] > 1)
        return Float{0};

    Float pdf = 1.0f / GetSurfaceArea();
    return pdf;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t Triangle<T>::SampleRNCount() const
{
    return 2;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Float Triangle<T>::GetSurfaceArea() const
{
    return ShapeFunctions::Triangle::Area(positions);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
AABB3 Triangle<T>::GetAABB() const
{
    return ShapeFunctions::Triangle::BoundingBox(positions);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Triangle<T>::GetCenter() const
{
    Vector3 center = (positions[0] * Float{0.333333} +
                      positions[1] * Float{0.333333} +
                      positions[2] * Float{0.333333});
    return center;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t Triangle<T>::Voxelize(Span<uint64_t>& mortonCodes,
                               Span<Vector2us>& normals,
                               bool onlyCalculateSize,
                               const VoxelizationParameters& voxelParams) const
{
    using namespace MathConstants;
    using namespace GraphicsFunctions;
    using Vector3::XAxis;
    using Vector3::YAxis;

    // Clang signbit definition is only on std namespace
    // this is a crappy workaround, since this is only a device function
    // but clang gives an error
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif

    // World Space Normal (Will be used to determine best projection plane)
    Vector3 normal = ShapeFunctions::Triangle::Normal(positions);
    normal = transformContext.ApplyN(normal);
    // Find the best projection plane (XY, YZ, XZ)
    int domAxis = normal.Abs().Max();
    bool hasNegSign = signbit(normal[domAxis]);
    float domSign = hasNegSign ? Float{-1} : Float{1};

    // Calculate Spherical UV Coordinates of the normal
    Vector2 sphrCoords = GraphicsFunctions::CartesianToUnitSpherical(normal);
    sphrCoords[0] = (sphrCoords[0] + Pi<Float>()) * InvPi<Float>() * Float{0.5};
    sphrCoords[1] = sphrCoords[1] * InvPi<Float>();
    // Due to numerical error this could slightly exceed [0, 65535]
    // clamp it
    Vector2i sphrUnormInt = Vector2i(static_cast<int32_t>(sphrCoords[0] * Float{65535}),
                                     static_cast<int32_t>(sphrCoords[1] * Float{65535}));
    sphrUnormInt.ClampSelf(0, 65535);
    Vector2us sphrUnorm = Vector2us(sphrUnormInt[0], sphrUnormInt[1]);

    // Look towards to the dominant axis
    Quaternion rot;
    switch(domAxis)
    {
        case 0: rot = Quaternion(Pi<Float>() * Float {0.5}, -domSign * YAxis()); break;
        case 1: rot = Quaternion(Pi<Float>() * Float {0.5},  domSign * XAxis()); break;
        case 2: rot = (hasNegSign) ? Quaternion(Pi<Float>(), YAxis()) : Quaternion::Identity(); break;
        default: assert(false); return 0;
    }
    // Generate a projection matrix (orthogonal)
    const AABB3 sceneAABB = voxelParams.sceneExtents;
    Matrix4x4 proj = TransformGen::Ortogonal(sceneAABB.Min()[0], sceneAABB.Max()[0],
                                             sceneAABB.Max()[1], sceneAABB.Min()[1],
                                             sceneAABB.Min()[2], sceneAABB.Max()[2]);

    // Apply Transformations
    Vector3 positionsT[3];
    // First Project
    positionsT[0] = Vector3(proj * Vector4(positions[0], 1.0f));
    positionsT[1] = Vector3(proj * Vector4(positions[1], 1.0f));
    positionsT[2] = Vector3(proj * Vector4(positions[2], 1.0f));
    // Now rotate towards Z axis
    positionsT[0] = rot.ApplyRotation(positionsT[0]);
    positionsT[1] = rot.ApplyRotation(positionsT[1]);
    positionsT[2] = rot.ApplyRotation(positionsT[2]);

    Vector2 positions2D[3];
    positions2D[0] = Vector2(positionsT[0]);
    positions2D[1] = Vector2(positionsT[1]);
    positions2D[2] = Vector2(positionsT[2]);

    // Finally Triangle is on NDC
    // Find AABB then start scan line
    Vector2 aabbMin = Vector2(std::numeric_limits<Float>::max());
    Vector2 aabbMax = Vector2(-std::numeric_limits<Float>::max());
    aabbMin = Vector2::Min(aabbMin, positions2D[0]);
    aabbMin = Vector2::Min(aabbMin, positions2D[1]);
    aabbMin = Vector2::Min(aabbMin, positions2D[2]);

    aabbMax = Vector2::Max(aabbMax, positions2D[0]);
    aabbMax = Vector2::Max(aabbMax, positions2D[1]);
    aabbMax = Vector2::Max(aabbMax, positions2D[2]);

    // Convert to [0, resolution] (pixel space)
    Float voxelResolution = Float{voxelParams.resolution};
    Vector2i xRangeInt(floor((Float{0.5} + Float{0.5} *aabbMin[0]) * voxelResolution),
                       ceil((Float{0.5} + Float{0.5} *aabbMax[0]) * voxelResolution));
    Vector2i yRangeInt(floor((Float{0.5} + Float{0.5} *aabbMin[1]) * voxelResolution),
                       ceil((Float{0.5} + Float{0.5} *aabbMax[1]) * voxelResolution));
    // Clip the range
    xRangeInt.ClampSelf(0, voxelParams.resolution);
    yRangeInt.ClampSelf(0, voxelParams.resolution);

    // Conservative Rasterization
    // Move all the edges "outwards" at least half a pixel
    // Notice NDC is [-1, 1] pixel size is 2 / resolution
    const float halfPixel = Float{1} / voxelResolution;
    const float deltaPix = Float{2} * halfPixel;
    // https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization
    // This was CG shader code which was optimized
    // with a single cross product you can find the line equation
    // ax + by + c = 0 (planes variable holds a,b,c)
    Vector3 planes[3];
    planes[0] = Vector3::Cross(Vector3(positions2D[1] - positions2D[0], 0),
                               Vector3(positions2D[0], 1));
    planes[1] = Vector3::Cross(Vector3(positions2D[2] - positions2D[1], 0),
                               Vector3(positions2D[1], 1));
    planes[2] = Vector3::Cross(Vector3(positions2D[0] - positions2D[2], 0),
                               Vector3(positions2D[2], 1));
    // Move the planes by the appropriate diagonal
    planes[0][2] -= Vector2(halfPixel).Dot(Vector2(planes[0]).Abs());
    planes[1][2] -= Vector2(halfPixel).Dot(Vector2(planes[1]).Abs());
    planes[2][2] -= Vector2(halfPixel).Dot(Vector2(planes[2]).Abs());
    // Compute the intersection point of the planes.
    // Again this code utilizes cross product to find x,y positions with (w)
    // which was implicitly divided by the rasterizer pipeline
    Vector3 positionsConserv[3];
    positionsConserv[0] = Vector3::Cross(planes[0], planes[2]);
    positionsConserv[1] = Vector3::Cross(planes[0], planes[1]);
    positionsConserv[2] = Vector3::Cross(planes[1], planes[2]);
    // Manually divide "w" (in this case Z) manually
    Vector2 positionsConsv2D[3];
    positionsConsv2D[0] = Vector2(positionsConserv[0]) / positionsConserv[0][2];
    positionsConsv2D[1] = Vector2(positionsConserv[1]) / positionsConserv[1][2];
    positionsConsv2D[2] = Vector2(positionsConserv[2]) / positionsConserv[2][2];

    // Generate Edges (for Cramer's Rule)
    // & iteration constants
    // Conservative Edges
    const Vector2 eCons0 = positionsConsv2D[1] - positionsConsv2D[0];
    const Vector2 eCons1 = positionsConsv2D[2] - positionsConsv2D[0];
    const float denomCons = Float{1} / (eCons0[0] * eCons1[1] - eCons1[0] * eCons0[1]);
    // Actual Edges
    const Vector2 e0 = positions2D[1] - positions2D[0];
    const Vector2 e1 = positions2D[2] - positions2D[0];
    const float denom = Float{1} / (e0[0] * e1[1] - e1[0] * e0[1]);
    // Scan Line Rasterization
    uint32_t writeIndex = 0;
    for(int y = yRangeInt[0]; y < yRangeInt[1]; y++)
    for(int x = xRangeInt[0]; x < xRangeInt[1]; x++)
    {
        // Gen Point (+0.5 for pixel middle)
        Vector2 pos = Vector2((static_cast<float>(x) + Float{0.5}) * deltaPix - Float{1},
                              (static_cast<float>(y) + Float{0.5}) * deltaPix - Float{1});

        // Cramer's Rule
        Vector2 eCons2 = pos - positionsConsv2D[0];
        float b = (eCons2[0] * eCons1[1] - eCons1[0] * eCons2[1]) * denomCons;
        float c = (eCons0[0] * eCons2[1] - eCons2[0] * eCons0[1]) * denomCons;
        float a = 1.0f - b - c;
        // If barycentrics are in range
        if(b >= 0 && b <= 1 &&
           c >= 0 && c <= 1 &&
           a >= 0 && a <= 1)
        {
            // Find the Actual Bary Coords here
            // Cramer's Rule
            Vector2 e2 = pos - positions2D[0];
            float actualB = (e2[0] * e1[1] - e1[0] * e2[1]) * denom;
            float actualC = (e0[0] * e2[1] - e2[0] * e0[1]) * denom;
            float actualA = 1 - actualB - actualC;

            // Bary's match, pixel is inside the triangle
            if(!onlyCalculateSize)
            {
                Vector3 voxelPos = (positions[0] * actualA +
                                    positions[1] * actualB +
                                    positions[2] * actualC);

                Vector3 voxelIndexF = ((voxelPos - sceneAABB.Min()) / sceneAABB.Span());
                voxelIndexF *= voxelResolution;
                Vector3ui voxelIndex = Vector3ui(static_cast<uint32_t>(voxelIndexF[0]),
                                                 static_cast<uint32_t>(voxelIndexF[1]),
                                                 static_cast<uint32_t>(voxelIndexF[2]));

                // TODO: This sometimes happen but it shouldn't??
                // Clamp the Voxel due to numerical errors
                voxelIndex.ClampSelf(0, voxelParams.resolution - 1);

                uint64_t voxelIndexMorton = MortonCode::Compose3D<uint64_t>(voxelIndex);
                // Write the found voxel
                assert(writeIndex < normals.size());
                assert(writeIndex < mortonCodes.size());
                assert(normals.size() == mortonCodes.size());
                if(writeIndex < mortonCodes.size())
                {
                    mortonCodes[writeIndex] = voxelIndexMorton;
                    normals[writeIndex] = sphrUnorm;
                }
            }
            writeIndex++;
        }
    }
    return writeIndex;
}


template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<BasicSurface> Triangle<T>::SurfaceFromHit(const Hit& hit) const
{
    Vector3 baryCoords = Vector3(hit[0], hit[1], 1 - hit[0] - hit[1]);
    if(baryCoords[0] < 0 || baryCoords[0] > 1 ||
       baryCoords[1] < 0 || baryCoords[1] > 1 ||
       baryCoords[2] < 0 || baryCoords[2] > 1)
        return std::nullopt;

    Vector3 position = (positions[0] * baryCoords[0] +
                        positions[1] * baryCoords[1] +
                        positions[2] * baryCoords[2]);

    Vector3ui index = data.indexList[id];
    Quaternion q0 = data.tbnRotations[index[0]];
    Quaternion q1 = data.tbnRotations[index[1]];
    Quaternion q2 = data.tbnRotations[index[2]];
    Quaternion tbn = Quaternion::BarySLerp(q0, q1, q2, baryCoords[0], baryCoords[1]);
    Vector3 normal = tbn.Conjugate().ApplyRotation(Vector3::ZAxis());

    return BasicSurface
    {
        .position = position,
        .normal = normal
    };
}


template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<TriHit> Triangle<T>::ProjectedHit(const Vector3& point) const
{
    using namespace ShapeFunctions::Triangle;
    Vector3 projPoint = Project(positions, point);
    Vector3 baryCoords = PointToBarycentrics(positions, projPoint);
    return Hit(baryCoords);
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 Triangle<T>::SurfaceParametrization(const Hit& hit) const
{
    Vector3ui index = data.indexList[id];
    Vector2 uv0 = data.uvs[index[0]];
    Vector2 uv1 = data.uvs[index[1]];
    Vector2 uv2 = data.uvs[index[2]];

    Vector3 baryCoords = Vector3(hit[0], hit[1], 1 - hit[1] - hit[0]);

    Vector2 uv = (baryCoords[0] * uv0 +
                  baryCoords[1] * uv1 +
                  baryCoords[2] * uv2);

    return uv;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Triangle<T>::GenerateSurface(EmptySurface&,
                                  // Inputs
                                  const Hit& baryCoords,
                                  const Ray& ray,
                                  const DiffRay& differentials) const
{}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Triangle<T>::GenerateSurface(BasicSurface& result,
                                  // Inputs
                                  const Hit& baryCoords,
                                  const Ray& ray,
                                  const DiffRay& differentials) const
{
    // Check if the prim is two-sided
    uint32_t primSubBatchId = data.subBatchTable.FindIndex(id);
    bool twoSided = !data.cullFace[primSubBatchId];

    Float a = baryCoords[0];
    Float b = baryCoords[1];
    Float c = Float{1} - a - b;

    // Get the position
    Vector3 pos = (positions[0] * a +
                   positions[1] * b +
                   positions[2] * c);
    // Position should be in world space
    using ShapeFunctions::Triangle::Normal;
    Vector3 geoNormal = Normal(positions);

    bool backSide = twoSided && (geoNormal.Dot(ray.Dir()) > 0.0f);
    if(backSide) geoNormal = -geoNormal;

    result = BasicSurface
    {
        .position = pos,
        .normal = geoNormal
    };
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Triangle<T>::GenerateSurface(BarycentricSurface& result,
                                  // Inputs
                                  const Hit& baryCoords,
                                  const Ray& ray,
                                  const DiffRay& differentials) const
{
    Float a = baryCoords[0];
    Float b = baryCoords[1];
    Float c = Float{1} - a - b;

    // Get the position
    Vector3 pos = (positions[0] * a +
                   positions[1] * b +
                   positions[2] * c);

    return BarycentricSurface
    {
        .position = pos,
        .baryCoords = Vector3(a, b, c)
    };
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
void Triangle<T>::GenerateSurface(DefaultSurface& result,
                                  // Inputs
                                  const Hit& baryCoords,
                                  const Ray& ray,
                                  const DiffRay& differentials) const
{
    static constexpr auto VPP = ShapeFunctions::Triangle::TRI_VERTEX_COUNT;

    // Check if the prim is two-sided
    uint32_t primSubBatchId = data.subBatchTable.FindIndex(id);
    bool twoSided = !data.cullFace[primSubBatchId];

    Float a = baryCoords[0];
    Float b = baryCoords[1];
    Float c = Float{1} - a - b;

    // Get the position
    Vector3 pos = (positions[0] * a +
                   positions[1] * b +
                   positions[2] * c);
    // Position should be in world space
    Vector3 geoNormal = ShapeFunctions::Triangle::Normal(positions);

    Vector3ui index = data.indexList[id];
    // Tangent Space Rotation Query
    Quaternion q0 = data.tbnRotations[index[0]];
    Quaternion q1 = data.tbnRotations[index[1]];
    Quaternion q2 = data.tbnRotations[index[2]];
    Quaternion tbn = Quaternion::BarySLerp(q0, q1, q2, a, b);
    tbn.NormalizeSelf();

    // If The requested primitive is two sided
    // Flip the surface definitions (normal, geometric normal)
    bool backSide = twoSided && (geoNormal.Dot(ray.Dir()) > 0.0f);
    if(backSide)
    {
        geoNormal = -geoNormal;
        // Change the worldToTangent rotation so that Z is on opposite direction
        // TODO: here flipping Z would change the handedness of the
        // coordinate system
        // Just adding the 180degree rotation with the tangent axis
        // to the end which should be fine I guess?
        static constexpr Quaternion TANGENT_ROT = Quaternion(0, 1, 0, 0);
        tbn = TANGENT_ROT * tbn;
    }

    // Transform to the requested space
    // pos already pre-transformed
    // geoNormal generated from pre-transformed pos's
    // we can't apply a transform to tbn
    result = DefaultSurface
    {
        .position = pos,
        .geoNormal = geoNormal,
        .shadingTBN = tbn,
        .backSide = backSide
    };
}

}

namespace DefaultSkinnedTriangleDetail
{

MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextSkinned::TransformContextSkinned(const typename TransformGroupMulti::DataSoA& transformData,
                                                 const SkinnedTriangleData& triData,
                                                 TransformId tId,
                                                 PrimitiveId pId)
{
    Vector4 weights = Vector4(triData.skinWeights[pId]);
    Vector4uc indices = triData.skinIndices[pId];
    const Span<const Matrix4x4>& t = transformData.transforms[tId];
    const Span<const Matrix4x4>& tInverse = transformData.invTransforms[tId];

    // Blend Transforms
    transform = Matrix4x4::Zero();
    UNROLL_LOOP
    for(int i = 0; i < TRANSFORM_PER_PRIMITIVE; i++)
    {
        transform += t[indices[i]] * weights[i];
    }

    // Blend Inverse Transforms
    invTransform = Matrix4x4::Zero();
    UNROLL_LOOP
    for(int i = 0; i < TRANSFORM_PER_PRIMITIVE; i++)
    {
        invTransform += tInverse[indices[i]] * weights[i];
    }
}

// Transform Context Generators
MRAY_HYBRID MRAY_CGPU_INLINE
TransformContextSkinned GenTContextSkinned(const typename TransformGroupMulti::DataSoA& tData,
                                           const SkinnedTriangleData& pData,
                                           TransformId tId,
                                           PrimitiveId pId)
{
    return TransformContextSkinned(tData, pData, tId, pId);
}

}