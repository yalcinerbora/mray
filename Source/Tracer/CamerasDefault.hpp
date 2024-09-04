#pragma once

namespace CameraDetail
{

MRAY_HYBRID MRAY_CGPU_INLINE
CameraPinhole::CameraPinhole(const DataSoA& soa, CameraKey key)
{
    position = soa.position[key.FetchIndexPortion()];
    up = soa.up[key.FetchIndexPortion()];
    Vector3  gaze = soa.gaze[key.FetchIndexPortion()];

    Vector4 fovAndPlanes = soa.fovAndPlanes[key.FetchIndexPortion()];
    nearFar = Vector2(fovAndPlanes[2], fovAndPlanes[3]);
    fov = Vector2(fovAndPlanes[0], fovAndPlanes[1]);

    // Find world space window sizes
    float widthHalf = tanf(fov[0] * Float(0.5)) * nearFar[0];
    float heightHalf = tanf(fov[1] * Float(0.5)) * nearFar[0];

    // Camera Vector Correction
    gazeDir = gaze - position;
    right = Vector3::Cross(gazeDir, up).Normalize();
    up = Vector3::Cross(right, gazeDir).Normalize();
    gazeDir = Vector3::Cross(up, right).Normalize();

    // Camera parameters
    bottomLeft = (position
                  - right * widthHalf
                  - up * heightHalf
                  + gazeDir * nearFar[0]);

    planeSize = Vector2(widthHalf, heightHalf) * Float(2.0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
RaySample CameraPinhole::SampleRay(// Input
                                   const Vector2ui& generationIndex,
                                   const Vector2ui& stratumCount,
                                   // I-O
                                   RNGDispenser& rng) const
{
    Vector2ui sampleId = generationIndex % stratumCount;

    // DX DY from stratified sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<Float>(stratumCount[0]),
                            planeSize[1] / static_cast<Float>(stratumCount[1]));

    // Create random location over sample rectangle
    Vector2 xi = rng.NextFloat2D<0>();

    Vector2 jitter = (xi * delta);
    Vector2 sampleDistance = Vector2(static_cast<float>(sampleId[0]),
                                     static_cast<float>(sampleId[1])) * delta;
    sampleDistance += jitter;
    Vector3 samplePoint = bottomLeft + ((sampleDistance[0] * right) +
                                        (sampleDistance[1] * up));
    Vector3 rayDir = (samplePoint - position).Normalize();

    // Local Coords
    // We are quantizing here, we probably have a validation
    // of images no larger than 65536. But this is here just to
    // be sure.
    assert(sampleId[0] <= std::numeric_limits<uint16_t>::max() &&
           sampleId[1] <= std::numeric_limits<uint16_t>::max());

    Vector2 localJitter = xi - Vector2(0.5);
    Vector2 out = Vector2(SNorm2x16(localJitter));
    ImageCoordinate imgCoords =
    {
        .pixelIndex = Vector2us(sampleId),
        .offset = SNorm2x16(localJitter)
    };
    // Initialize Ray
    return RaySample
    {
        .value =
        {
            .ray = Ray(rayDir, position),
            .tMinMax = nearFar,
            .imgCoords = imgCoords,
            .rayDifferentials = RayDiff{}
        },
        .pdf = Float(1.0)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float CameraPinhole::PdfRay(const Ray&) const
{
    // We can not request pdf of a pinhole camera
    return Float(0.0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
bool CameraPinhole::CanBeSampled() const
{
    return false;
}

MRAY_HYBRID MRAY_CGPU_INLINE
CameraTransform CameraPinhole::GetCameraTransform() const
{
    Vector3 dir = Vector3::Cross(up, right).Normalize();
    return CameraTransform
    {
        .position   = position,
        .gazePoint  = position + dir,
        .up         = up
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
void CameraPinhole::OverrideTransform(const CameraTransform& t)
{
    position = t.position;
    up = t.up;
    Vector3 gazePoint = t.gazePoint;

    // Camera Vector Correction
    gazeDir = gazePoint - position;
    right = Vector3::Cross(gazeDir, up).Normalize();
    up = Vector3::Cross(right, gazeDir).Normalize();
    gazeDir = Vector3::Cross(up, right).Normalize();

    // Camera parameters
    float widthHalf = planeSize[0] * Float(0.5);
    float heightHalf = planeSize[1] * Float(0.5);

    bottomLeft = (position
                  - right * widthHalf
                  - up * heightHalf
                  + gazeDir * nearFar[0]);
}

MRAY_HYBRID MRAY_CGPU_INLINE
CameraPinhole CameraPinhole::GenerateSubCamera(const Vector2ui& statumIndex,
                                               const Vector2ui& stataCount) const
{
    // DX DY from stratified sample
    Vector2 delta = Vector2(planeSize[0] / static_cast<Float>(stataCount[0]),
                            planeSize[1] / static_cast<Float>(stataCount[1]));

    Vector2 regionlDistance = Vector2(static_cast<Float>(statumIndex[0]),
                                      static_cast<Float>(statumIndex[1])) * delta;
    Vector3 regionBottomLeft = bottomLeft + ((regionlDistance[0] * right) +
                                             (regionlDistance[1] * up));

    Vector2 fovRegion = Vector2(fov[0] / stataCount[0],
                                fov[1] / stataCount[1]);

    CameraPinhole p = CameraPinhole(*this);
    // Change the bottom left, and the FoV
    p.fov = fovRegion;
    p.planeSize = delta;
    p.bottomLeft = regionBottomLeft;
    return p;
}

}