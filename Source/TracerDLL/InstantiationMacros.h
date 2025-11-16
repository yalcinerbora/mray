#pragma once

#include "Tracer/PrimitiveC.h"

#include <cstdint>

// These are for instantiation convenience
// Work functions has quite a bit for template parameters
// ===================== //
//  Get Work Metaprogram //
// ===================== //
template <class R, class PG, class MG, class TG, uint32_t I>
struct GetWFT
{
    private:
    using WFList = R::template WorkFunctions<PG, MG, TG>;
    public:
    using Type = std::conditional_t<I < WFList::TypeCount, TypePackElement<I, WFList>, void>;
};

template <class R, class PG, class MG, class TG, uint32_t I>
using GetWF = GetWFT<R, PG, MG, TG, I>::Type;

// =========================== //
//  Get Light Work Metaprogram //
// =========================== //
template <class R, class LG, class TG, uint32_t I>
struct GetLWFT
{
    private:
    using WFList = R::template LightWorkFunctions<LG, TG>;
    public:
    using Type = std::conditional_t<I < WFList::TypeCount, TypePackElement<I, WFList>, void>;
};

template <class R, class LG, class TG, uint32_t I>
using GetLWF = GetLWFT<R, LG, TG, I>::Type;

// ==========================//
//  Get Cam Work Metaprogram //
// ========================= //
template <class R, class CG, class TG, uint32_t I>
struct GetCWFT
{
    private:
    using WFList = R::template CamWorkFunctions<CG, TG>;
    public:
    using Type = std::conditional_t<I < WFList::TypeCount, TypePackElement<I, WFList>, void>;
};
template <class R, class CG, class TG, uint32_t I>
using GetCWF = GetCWFT<R, CG, TG, I>::Type;

// ==========================//
//  Get Cam Work Metaprogram //
// ========================= //
template <class R, class MG, class TG, uint32_t I>
struct GetMWFT
{
    private:
    using WFList = R::template MediaWorkFunctions<MG, TG>;
    public:
    using Type = std::conditional_t<I < WFList::TypeCount, TypePackElement<I, WFList>, void>;
};
template <class R, class MG, class TG, uint32_t I>
using GetMWF = GetMWFT<R, MG, TG, I>::Type;


// Renderer Related
#define MRAY_RENDERER_KERNEL_INSTANTIATE(R, P, M, T, I)                \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT             \
    void KCRenderWork                                                  \
    <                                                                  \
        GetWF<R, P, M, T, I>,                                          \
        AcquireSpectrumConverterGenerator<R, GetWF<R, P, M, T, I>>(),  \
        MRAY_PRIM_TGEN_FUNCTION(P, T)                                  \
    >                                                                  \
    (MRAY_GRID_CONSTANT const typename GetWF<R, P, M, T, I>::Params)


#define MRAY_RENDERER_LIGHT_KERNEL_INSTANTIATE(R, L, T, I)          \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT          \
    void KCRenderLightWork                                          \
    <                                                               \
        GetLWF<R, L, T, I>,                                         \
        AcquireSpectrumConverterGenerator<R, GetLWF<R, L, T, I>>(), \
        MRAY_PRIM_TGEN_FUNCTION(typename L::PrimGroup, T)           \
    >                                                               \
    (MRAY_GRID_CONSTANT const typename GetLWF<R, L, T, I>::Params)

#define MRAY_RENDERER_CAM_KERNEL_INSTANTIATE(R, C, T, I)            \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT          \
    void KCRenderCameraWork                                         \
    <                                                               \
        GetCWF<R, C, T, I>                                          \
    >                                                               \
    (MRAY_GRID_CONSTANT const typename GetCWF<R, C, T, I>::Params)

#define MRAY_RENDERER_MEDIUM_KERNEL_INSTANTIATE(R, M, T, I)         \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT          \
    void KCRenderMediumWork                                         \
    <                                                               \
        GetMWF<R, C, T, I>,                                         \
        AcquireSpectrumConverterGenerator<R, GetMWF<R, L, T, I>>(), \
        AcquireTransformContextGenerator<PrimGroupEmpty, TG>()      \
    >                                                               \
    (MRAY_GRID_CONSTANT const typename GetMWF<R, C, T, I>::Params)

// Accelerator Related
#define MRAY_ACCEL_PRIM_CENTER_KERNEL_INSTANTIATE(A, P, T)                                                              \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                                              \
    void KCGenPrimCenters<A<P>, T, MRAY_ACCEL_TGEN_FUNCTION(A<P>, T)>(MRAY_GRID_CONSTANT const Span<Vector3>,           \
                                                                      MRAY_GRID_CONSTANT const Span<const uint32_t>,    \
                                                                      MRAY_GRID_CONSTANT const Span<const TransformKey>,\
                                                                      MRAY_GRID_CONSTANT const Span<const PrimitiveKey>,\
                                                                      MRAY_GRID_CONSTANT const uint32_t,                \
                                                                      MRAY_GRID_CONSTANT const uint32_t,                \
                                                                      MRAY_GRID_CONSTANT const typename T::DataSoA,     \
                                                                      MRAY_GRID_CONSTANT const typename P::DataSoA)

#define MRAY_ACCEL_PRIM_AABB_KERNEL_INSTANTIATE(A, P, T)                                                                    \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                                                  \
    void KCGeneratePrimAABBs<A<P>, T, MRAY_ACCEL_TGEN_FUNCTION(A<P>, T)>(MRAY_GRID_CONSTANT const Span<AABB3>,              \
                                                                         MRAY_GRID_CONSTANT const Span<const uint32_t>,     \
                                                                         MRAY_GRID_CONSTANT const Span<const TransformKey>, \
                                                                         MRAY_GRID_CONSTANT const Span<const PrimitiveKey>, \
                                                                         MRAY_GRID_CONSTANT const uint32_t,                 \
                                                                         MRAY_GRID_CONSTANT const uint32_t,                 \
                                                                         MRAY_GRID_CONSTANT const typename T::DataSoA,      \
                                                                         MRAY_GRID_CONSTANT const typename P::DataSoA)

#define MRAY_ACCEL_COMMON_TRANSFORM_KERNEL_INSTANTIATE(T)                           \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                          \
    void KCGetCommonTransforms<T>(MRAY_GRID_CONSTANT const Span<Matrix4x4>,         \
                                  MRAY_GRID_CONSTANT const Span<const TransformKey>,\
                                  MRAY_GRID_CONSTANT const typename T::DataSoA)

#define MRAY_ACCEL_TRANSFORM_AABB_KERNEL_INSTANTIATE(A, P, T)                                                                           \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                                                              \
    void KCTransformLocallyConstantAABBs<A<P>, T, MRAY_ACCEL_TGEN_FUNCTION(A<P>, T)>(MRAY_GRID_CONSTANT const Span<AABB3>,              \
                                                                                     MRAY_GRID_CONSTANT const Span<const AABB3>,        \
                                                                                     MRAY_GRID_CONSTANT const Span<const uint32_t>,     \
                                                                                     MRAY_GRID_CONSTANT const Span<const TransformKey>, \
                                                                                     MRAY_GRID_CONSTANT const typename T::DataSoA,      \
                                                                                     MRAY_GRID_CONSTANT const typename P::DataSoA)

#define MRAY_ACCEL_LOCAL_RAY_CAST_KERNEL_INSTANTIATE(A, P, T)                                                               \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                                                  \
    void KCLocalRayCast<A<P>, T, MRAY_ACCEL_TGEN_FUNCTION(A<P>, T)>(MRAY_GRID_CONSTANT const Span<InterfaceIndex>,          \
                                                                    MRAY_GRID_CONSTANT const Span<HitKeyPack>,              \
                                                                    MRAY_GRID_CONSTANT const Span<MetaHit>,                 \
                                                                    MRAY_GRID_CONSTANT const Span<BackupRNGState>,          \
                                                                    MRAY_GRID_CONSTANT const Span<RayGMem>,                 \
                                                                    MRAY_GRID_CONSTANT const Span<const RayIndex>,          \
                                                                    MRAY_GRID_CONSTANT const Span<const CommonKey>,         \
                                                                    MRAY_GRID_CONSTANT const typename T::DataSoA,           \
                                                                    MRAY_GRID_CONSTANT const typename A<P>::DataSoA,        \
                                                                    MRAY_GRID_CONSTANT const typename P::DataSoA,           \
                                                                    MRAY_GRID_CONSTANT const bool)

#define MRAY_ACCEL_VISIBILITY_RAY_CAST_KERNEL_INSTANTIATE(A, P, T)                                                          \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                                                  \
    void KCVisibilityRayCast<A<P>, T, MRAY_ACCEL_TGEN_FUNCTION(A<P>, T)>(MRAY_GRID_CONSTANT const Bitspan<uint32_t>,        \
                                                                         MRAY_GRID_CONSTANT const Span<BackupRNGState>,     \
                                                                         MRAY_GRID_CONSTANT const Span<const RayGMem>,      \
                                                                         MRAY_GRID_CONSTANT const Span<const RayIndex>,     \
                                                                         MRAY_GRID_CONSTANT const Span<const CommonKey>,    \
                                                                         MRAY_GRID_CONSTANT const typename T::DataSoA,      \
                                                                         MRAY_GRID_CONSTANT const typename A<P>::DataSoA,   \
                                                                         MRAY_GRID_CONSTANT const typename P::DataSoA)
// Camera Related
#define MRAY_RAYGEN_SUBCAMERA_KERNEL_INSTANTIATE(C, T)                                  \
    template MRAY_KERNEL                                                                \
    void KCGenerateSubCamera<C, T>(MRAY_GRID_CONSTANT typename C::Camera* const,        \
                                   MRAY_GRID_CONSTANT const CameraKey,                  \
                                   MRAY_GRID_CONSTANT const typename C::DataSoA,        \
                                   MRAY_GRID_CONSTANT const Optional<CameraTransform>,  \
                                   MRAY_GRID_CONSTANT const Vector2ui,                  \
                                   MRAY_GRID_CONSTANT const Vector2ui)

#define MRAY_RAYGEN_CAMERA_POS_KERNEL_INSTANTIATE(C, T)                                                      \
    template MRAY_KERNEL                                                                                     \
    void KCGenerateCameraPosition<typename C::Camera, T>(MRAY_GRID_CONSTANT const Span<Vector3, 1>,          \
                                                         MRAY_GRID_CONSTANT const typename C::Camera* const, \
                                                         MRAY_GRID_CONSTANT const TransformKey,              \
                                                         MRAY_GRID_CONSTANT const typename T::DataSoA)

#define MRAY_RAYGEN_RECONSTRUCT_RAYS_KERNEL_INSTANTIATE(C, T)                                                \
    template MRAY_KERNEL                                                                                     \
    void KCReconstructCameraRays<typename C::Camera, T>(MRAY_GRID_CONSTANT const Span<RayCone>,              \
                                                        MRAY_GRID_CONSTANT const Span<RayGMem>,              \
                                                        MRAY_GRID_CONSTANT const Span<const ImageCoordinate>,\
                                                        MRAY_GRID_CONSTANT const Span<const uint32_t>,       \
                                                        MRAY_GRID_CONSTANT const typename C::Camera* const,  \
                                                        MRAY_GRID_CONSTANT const TransformKey,               \
                                                        MRAY_GRID_CONSTANT const typename T::DataSoA,        \
                                                        MRAY_GRID_CONSTANT const Vector2ui)

#define MRAY_RAYGEN_GENRAYS_KERNEL_INSTANTIATE(C, T)                                                  \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                            \
    void KCGenerateCamRays<typename C::Camera, T>(MRAY_GRID_CONSTANT const Span<RayCone>,             \
                                                  MRAY_GRID_CONSTANT const Span<RayGMem>,             \
                                                  MRAY_GRID_CONSTANT const Span<ImageCoordinate>,     \
                                                  MRAY_GRID_CONSTANT const Span<Float>,               \
                                                  MRAY_GRID_CONSTANT const Span<const uint32_t>,      \
                                                  MRAY_GRID_CONSTANT const Span<const RandomNumber>,  \
                                                  MRAY_GRID_CONSTANT const typename C::Camera* const, \
                                                  MRAY_GRID_CONSTANT const TransformKey,              \
                                                  MRAY_GRID_CONSTANT const typename T::DataSoA,       \
                                                  MRAY_GRID_CONSTANT const uint64_t,                  \
                                                  MRAY_GRID_CONSTANT const Vector2ui)

#define MRAY_RAYGEN_GENRAYS_STOCHASTIC_KERNEL_INSTANTIATE(C, T, F)                                                  \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                                          \
    void KCGenerateCamRaysStochastic<typename C::Camera, T, F>(MRAY_GRID_CONSTANT const Span<RayCone>,              \
                                                               MRAY_GRID_CONSTANT const Span<RayGMem>,              \
                                                               MRAY_GRID_CONSTANT const Span<ImageCoordinate>,      \
                                                               MRAY_GRID_CONSTANT const Span<Float>,                \
                                                               MRAY_GRID_CONSTANT const Span<const uint32_t>,       \
                                                               MRAY_GRID_CONSTANT const Span<const RandomNumber>,   \
                                                               MRAY_GRID_CONSTANT const C::Camera* const,           \
                                                               MRAY_GRID_CONSTANT const TransformKey,               \
                                                               MRAY_GRID_CONSTANT const typename T::DataSoA,        \
                                                               MRAY_GRID_CONSTANT const uint64_t,                   \
                                                               MRAY_GRID_CONSTANT const Vector2ui,                  \
                                                               MRAY_GRID_CONSTANT const F)
