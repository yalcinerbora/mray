#pragma once

#include "RayGenKernels.h"

template<CameraGroupC CameraG, TransformGroupC TransG>
requires(ImplicitLifetimeC<typename CameraG::Camera>)
MRAY_KERNEL
void KCGenerateSubCamera(// Output
                         MRAY_GRID_CONSTANT
                         typename CameraG::Camera* const dCam,
                         // Constants
                         MRAY_GRID_CONSTANT const CameraKey camKey,
                         MRAY_GRID_CONSTANT const typename CameraG::DataSoA camSoA,
                         MRAY_GRID_CONSTANT const Optional<CameraTransform> camTransform,
                         MRAY_GRID_CONSTANT const Vector2ui stratumIndex,
                         MRAY_GRID_CONSTANT const Vector2ui stratumCount)
{
    using Camera = typename CameraG::Camera;
    KernelCallParams kp;
    if(kp.GlobalId() != 0) return;
    // Construction
    Camera cam = Camera(camSoA, camKey);
    if(camTransform.has_value())
        cam.OverrideTransform(camTransform.value());
    // Generate sub camera for rendered regions
    *dCam = cam.GenerateSubCamera(stratumIndex,
                                  stratumCount);
}

template<CameraC Camera, TransformGroupC TransG>
MRAY_KERNEL
void KCGenerateCameraPosition(// Output
                              MRAY_GRID_CONSTANT const Span<Vector3, 1> dCameraPosition,
                              // Input
                              MRAY_GRID_CONSTANT const Camera* const dCamera,
                              MRAY_GRID_CONSTANT const TransformKey transformKey,
                              MRAY_GRID_CONSTANT const typename TransG::DataSoA transSoA)
{
    KernelCallParams kp;
    if(kp.GlobalId() != 0) return;

    using TransformContext = TransG::DefaultContext;
    TransformContext transform(transSoA, transformKey);

    // Construction
    Vector3 localCamPos = dCamera->GetCameraPosition();
    Vector3 worldCamPos = transform.ApplyP(localCamPos);
    dCameraPosition[0] = worldCamPos;
}

template<CameraC Camera, TransformGroupC TransG>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateCamRays(// Output (Only dRayIndices pointed data should be written)
                       MRAY_GRID_CONSTANT const Span<RayCone> dRayCones,
                       MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                       MRAY_GRID_CONSTANT const Span<ImageCoordinate> dImageCoordinates,
                       MRAY_GRID_CONSTANT const Span<Float> dFilmFilterWeights,
                       // Input
                       MRAY_GRID_CONSTANT const Span<const uint32_t> dRayIndices,
                       MRAY_GRID_CONSTANT const Span<const RandomNumber> dRandomNums,
                       // Constants
                       MRAY_GRID_CONSTANT const Camera* const dCamera,
                       MRAY_GRID_CONSTANT const TransformKey transformKey,
                       MRAY_GRID_CONSTANT const typename TransG::DataSoA transSoA,
                       MRAY_GRID_CONSTANT const uint64_t globalRegionIndex,
                       MRAY_GRID_CONSTANT const Vector2ui regionCount)
{
    [[maybe_unused]]
    static constexpr auto RNPerCamSample = Camera::SampleRayRNList.TotalRNCount();
    assert(dRayIndices.size() * RNPerCamSample == dRandomNums.size());

    KernelCallParams kp;
    const Camera& dCam = *dCamera;

    // Get the transform
    using TransformContext = TransG::DefaultContext;
    TransformContext transform(transSoA, transformKey);

    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t globalId = kp.GlobalId(); globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(dRandomNums, globalId, rayCount);
        //
        uint64_t regionIndex = globalRegionIndex + globalId;
        uint64_t totalRegions = regionCount.Multiply();
        // Rollover, globalRegionIndex is never rolled to zero
        // (it is ever increasing, we may use it to show timings etc.)
        // so roll here (some other code will add rayCount to globalRegionIndex)
        regionIndex %= totalRegions;
        Vector2ui regionIndex2D = Vector2ui(regionIndex % regionCount[0],
                                            regionIndex / regionCount[0]);
        // Generate the sample
        CameraRaySample raySample = dCam.SampleRay(regionIndex2D,
                                                   regionCount, rng);
        // TODO: Should we normalize and push the length to tminmax
        // (Because of a scale?)
        raySample.value.ray = transform.Apply(raySample.value.ray);
        // Ray part is easy, just write
        uint32_t writeIndex = dRayIndices[globalId];
        RayToGMem(dRays, writeIndex,
                  raySample.value.ray,
                  raySample.value.tMinMax);

        // Write the differentials
        assert(Math::IsFinite(raySample.value.rayCone.aperture) &&
               Math::IsFinite(raySample.value.rayCone.width));
        dRayCones[writeIndex] = raySample.value.rayCone;

        // Finally write
        dImageCoordinates[writeIndex] = raySample.value.imgCoords;
        dFilmFilterWeights[writeIndex] = raySample.pdf;
    }
}

template<CameraC Camera, TransformGroupC TransG>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCReconstructCameraRays(// Output (Only dRayIndices pointed data should be written)
                             MRAY_GRID_CONSTANT const Span<RayCone> dRayCones,
                             MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                             // Input (only accessed via dRayIndices)
                             MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImageCoordinates,
                             MRAY_GRID_CONSTANT const Span<const uint32_t> dRayIndices,
                             // Constants
                             MRAY_GRID_CONSTANT const Camera* const dCamera,
                             MRAY_GRID_CONSTANT const TransformKey transformKey,
                             MRAY_GRID_CONSTANT const typename TransG::DataSoA transSoA,
                             MRAY_GRID_CONSTANT const Vector2ui stratumCount)
{
    KernelCallParams kp;
    const Camera& dCam = *dCamera;

    // Get the transform
    using TransformContext = TransG::DefaultContext;
    TransformContext transform(transSoA, transformKey);

    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t globalId = kp.GlobalId(); globalId < rayCount; globalId += kp.TotalSize())
    {
        RayIndex rIndex = dRayIndices[globalId];
        ImageCoordinate imgCoord = dImageCoordinates[rIndex];

        CameraRayOutput rayOut = dCam.ReconstructRay(imgCoord, stratumCount);
        // TODO: Should we normalize and push the length to tminmax
        // (Because of a scale?)
        rayOut.ray = transform.Apply(rayOut.ray);
        // Ray part is easy, just write
        RayToGMem(dRays, rIndex, rayOut.ray, rayOut.tMinMax);
        // Write the differentials
        assert(Math::IsFinite(rayOut.rayCone.aperture) &&
               Math::IsFinite(rayOut.rayCone.width));
        dRayCones[rIndex] = rayOut.rayCone;
    }
}

template<CameraC Camera, TransformGroupC TransG, class FilterType>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateCamRaysStochastic(// Output (Only dRayIndices pointed data should be written)
                                 MRAY_GRID_CONSTANT const Span<RayCone> dRayCones,
                                 MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                                 MRAY_GRID_CONSTANT const Span<ImageCoordinate> dImageCoordinates,
                                 MRAY_GRID_CONSTANT const Span<Float> dFilmFilterWeights,
                                 // Input
                                 MRAY_GRID_CONSTANT const Span<const uint32_t> dRayIndices,
                                 MRAY_GRID_CONSTANT const Span<const RandomNumber> dRandomNums,
                                 // Constants
                                 MRAY_GRID_CONSTANT const Camera* const dCamera,
                                 MRAY_GRID_CONSTANT const TransformKey transformKey,
                                 MRAY_GRID_CONSTANT const typename TransG::DataSoA transSoA,
                                 MRAY_GRID_CONSTANT const uint64_t globalRegionIndex,
                                 MRAY_GRID_CONSTANT const Vector2ui regionCount,
                                 MRAY_GRID_CONSTANT const FilterType Filter)
{
    [[maybe_unused]]
    static constexpr auto RNPerCamSample = Camera::SampleRayRNList.TotalRNCount();
    assert(dRayIndices.size() * RNPerCamSample == dRandomNums.size());

    KernelCallParams kp;
    const Camera& dCam = *dCamera;

    // Get the transform
    using TransformContext = TransG::DefaultContext;
    TransformContext transform(transSoA, transformKey);

    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t globalId = kp.GlobalId(); globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(dRandomNums, globalId, rayCount);
        //
        uint64_t regionIndex = globalRegionIndex + globalId;
        uint64_t totalRegions = regionCount.Multiply();
        // Rollover, globalRegionIndex is never rolled to zero
        // (it is ever increasing, we may use it to show timings etc.)
        // so roll here (some other code will add rayCount to globalRegionIndex)
        regionIndex %= totalRegions;
        Vector2ui regionIndex2D = Vector2ui(regionIndex % regionCount[0],
                                            regionIndex / regionCount[0]);

        Vector2 xi = rng.NextFloat2D<0>();
        SampleT<Vector2> offsetSample = Filter.Sample(xi);
        Float weight = Filter.Evaluate(offsetSample.value);

        // Evaluate the sample
        Float filterRadius = Filter.Radius();
        CameraRaySample raySample = dCam.EvaluateRay(regionIndex2D, regionCount,
                                                     offsetSample.value,
                                                     Vector2(filterRadius * Float(2)));
        // TODO: Should we normalize and push the length to tminmax
        // (Because of a scale?)
        raySample.value.ray = transform.Apply(raySample.value.ray);
        // Ray part is easy, just write
        uint32_t writeIndex = dRayIndices[globalId];
        RayToGMem(dRays, writeIndex,
                  raySample.value.ray,
                  raySample.value.tMinMax);

        // Write the differentials
        assert(Math::IsFinite(raySample.value.rayCone.aperture) &&
               Math::IsFinite(raySample.value.rayCone.width));
        dRayCones[writeIndex] = raySample.value.rayCone;
        // Write filter weights (pdf normalized)
        raySample.pdf = weight / offsetSample.pdf;

        // Finally write
        dImageCoordinates[writeIndex] = raySample.value.imgCoords;
        dFilmFilterWeights[writeIndex] = raySample.pdf;
    }
}