#pragma once

#include "CameraC.h"
#include "RendererC.h"
#include "Random.h"

template<CameraGroupC CameraG, TransformGroupC TransG>
// Camera must be implicit lifetime type.
// If some dynamic polymorphism kicks in somewhere in the code
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
                         MRAY_GRID_CONSTANT const Vector2ui stratumCount);

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
                       MRAY_GRID_CONSTANT const Vector2ui regionCount);

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
                                 MRAY_GRID_CONSTANT const FilterType Filter);