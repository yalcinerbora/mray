#pragma once

#include "CameraC.h"
#include "RendererC.h"
#include "Random.h"


template<CameraGroupC CameraG, TransformGroupC TransG>
// Camera must be implicit lifetime type.
// If some dynamic polymorphism kicksin somewhere in the code
// we will need to construct the camera via inplace new to create
// proper virtual function pointers (that refers the device code)
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

template<auto RayStateInitFunc,
         class RayPayload, class RayState,
         CameraC Camera, TransformGroupC TransG>
MRAY_KERNEL
void KCGenerateCamRays(// Output (Only dOutIndices pointed data should be written)
                       MRAY_GRID_CONSTANT const Span<RayDiff> dRayDiffs,
                       MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                       MRAY_GRID_CONSTANT const RayPayload dRayPayloads,
                       MRAY_GRID_CONSTANT const RayState dRayStates,
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
    assert(dRayIndices.size() * Camera::SampleRayRNCount == dRandomNums.size());
    KernelCallParams kp;
    const Camera& dCam = *dCamera;

    // Get the transform
    using TransformContext = TransG::DefaultContext;
    TransformContext transform(transSoA, transformKey);

    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t globalId = kp.GlobalId(); globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(dRandomNums, kp.GlobalId(), globalId);
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
        RaySample raySample = dCam.SampleRay(regionIndex2D, regionCount, rng);
        // TODO: Should we normalize and push the length to tminmax
        // (Because of a scale?)
        raySample.value.ray = transform.Apply(raySample.value.ray);
        // Ray part is easy, just write
        uint32_t writeIndex = dRayIndices[globalId];
        RayToGMem(dRays, writeIndex,
                  raySample.value.ray,
                  raySample.value.tMinMax);

        // Write the differentials
        dRayDiffs[writeIndex] = raySample.value.rayDifferentials;

        // Now we have stratification index (img coords if
        // stratification is 1 pixel) and pdf,
        // we pass it to the renderer .
        // For example;
        // A path tracer will save them directly and set the throughput to the pdf maybe
        RayStateInitFunc(dRayPayloads, dRayStates, writeIndex, raySample);
    }
}