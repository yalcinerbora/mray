#pragma once

#include "CameraC.h"
#include "RendererC.h"
#include "Random.h"


template<CameraC Camera>
// Camera must be implicit lifetime type.
// If some dynamic polymorphism kicksin somewhere in the code
// we will need to construct the camera via inplace new to create
// proper virtual function pointers (that refers the device code)
requires(ImplicitLifetimeC<Camera>)
MRAY_KERNEL
void KCGenerateSubCamera(// I-O
                         MRAY_GRID_CONSTANT const Camera* dCam,
                         //
                         MRAY_GRID_CONSTANT const typename Camera::DataSoA camSoA,
                         MRAY_GRID_CONSTANT const CameraKey camKey,
                         MRAY_GRID_CONSTANT const Optional<CameraTransform> camTransform,
                         MRAY_GRID_CONSTANT const Vector2ui stratumIndex,
                         MRAY_GRID_CONSTANT const Vector2ui stratumCount)
{
    if(kp.GlobalId() != 0) return;
    // Construction
    dCam = Camera(camSoA, camKey);
    if(camTransform.has_value())
        dCam = dCam.OverrideTransform(camTransform.value());
    // Generate sub camera for rendered regions
    dCam = dCam.GenerateSubCamera(stratumIndex,
                                stratumCount);
}

template<CameraC Camera, RendererC Renderer>
MRAY_KERNEL
void KCGenerateCamRays(// Output (Only dOutIndices pointed data should be written)
                       MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                       MRAY_GRID_CONSTANT const Span<typename Renderer::RayState> dRayState,
                       // Input
                       MRAY_GRID_CONSTANT const Span<const uint32_t> dOutIndices,
                       MRAY_GRID_CONSTANT const Span<const uint32_t> dRandomNums,
                       // Constants
                       MRAY_GRID_CONSTANT const Camera* dCamera,
                       MRAY_GRID_CONSTANT const uint64_t globalPixelIndex,
                       MRAY_GRID_CONSTANT const Vector2ui regionCount)
{
    assert(dOutIndices.size() * dCamera->SampleRayRNCount() <= dRandomNums.size())
    KernelCallParams kp;
    const Camera& dCam = *dCamera;

    uint32_t rayCount = static_cast<uint32_t>(dOutIndices.size());
    for(uint32_t globalId = kp.GlobalId(); globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(numbers, kp.GlobalId(), globalId);
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
        RaySample raySample = sCam.SampleRay(regionIndex2D, regionCount, rng);
        // Ray part is easy, just write
        uint32_t writeIndex = dOutIndices[globalId];
        RayToGMem(dRays, writeIndex,
                  raySample.value.ray,
                  raySample.value.tMinMax);
        // Now we have stratification index (img coords if
        // stratification is 1 pixel) and pdf,
        // we pass it to the renderer .
        // For example;
        // A path tracer will save them directly and set the throughput to the pdf maybe
        dRayStates[writeIndex] = Renderer::InitRayState(raySample);
    }
}