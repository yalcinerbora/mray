#pragma once

#include "Tracer/PrimitiveC.h"

#include <cstdint>

// These are for instantiation convenience
// Work functions has quite a bit for template parameters
template <class R, class PG, class MG, class TG, uint32_t I>
static constexpr auto GetWorkFunction()
{
    {
        using TC = typename PrimTransformContextType<PG, TG>::Result;
        using M = typename MG:: template Material<typename R::SpectrumContext>;
        using P = typename PG:: template Primitive<TC>;
        using S = typename M::Surface;
        constexpr auto WF = R::template WorkFunctions<P, M, S, TC, PG, MG, TG>;
        return get<I>(WF);
    }
};
template <class R, class LG, class TG, uint32_t I>
static constexpr auto GetLightWorkFunction()
{
    {
        using PG = typename LG::PrimGroup;
        using TC = typename PrimTransformContextType<PG, TG>::Result;
        using L = typename LG::template Light<TC, typename R::SpectrumContext>;
        constexpr auto WF = R:: template LightWorkFunctions<L, LG, TG>;
        return get<I>(WF);
    }
};
template <class R, class CG, class TG, uint32_t I>
static constexpr auto GetCamWorkFunction()
{
    {
        using C = typename CG::Camera;
        constexpr auto WF = R:: template CamWorkFunctions<C, CG, TG>;
        return get<I>(WF);
    }
};

// Renderer Related
#define MRAY_RENDERER_KERNEL_INSTANTIATE(R, P, M, T, I)     \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT  \
    void KCRenderWork<R, I, P, M, T, GetWorkFunction<R, P, M, T, I>(), MRAY_PRIM_TGEN_FUNCTION(P, T)>(MRAY_GRID_CONSTANT const RenderWorkParamsR<R, I, P, M, T>)

#define MRAY_RENDERER_LIGHT_KERNEL_INSTANTIATE(R, L, T, I)  \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT  \
    void KCRenderLightWork<R, I, L, T, GetLightWorkFunction<R, L, T, I>(), MRAY_PRIM_TGEN_FUNCTION(typename L::PrimGroup, T)>(MRAY_GRID_CONSTANT const RenderLightWorkParamsR<R, I, L, T>)

#define MRAY_RENDERER_CAM_KERNEL_INSTANTIATE(R, C, T, I)    \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT  \
    void KCRenderCameraWork<R, I, C, T, GetCamWorkFunction<R, C, T, I>()>(MRAY_GRID_CONSTANT const RenderCameraWorkParamsR<R, I, C, T>)

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
    void KCLocalRayCast<A<P>, T, MRAY_ACCEL_TGEN_FUNCTION(A<P>, T)>(MRAY_GRID_CONSTANT const Span<HitKeyPack>,              \
                                                                    MRAY_GRID_CONSTANT const Span<MetaHit>,                 \
                                                                    MRAY_GRID_CONSTANT const Span<BackupRNGState>,          \
                                                                    MRAY_GRID_CONSTANT const Span<RayGMem>,                 \
                                                                    MRAY_GRID_CONSTANT const Span<const RayIndex>,          \
                                                                    MRAY_GRID_CONSTANT const Span<const CommonKey>,         \
                                                                    MRAY_GRID_CONSTANT const typename T::DataSoA,           \
                                                                    MRAY_GRID_CONSTANT const typename A<P>::DataSoA,        \
                                                                    MRAY_GRID_CONSTANT const typename P::DataSoA)

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

#define MRAY_RAYGEN_GENRAYS_KERNEL_INSTANTIATE(C, T)                                                \
    template MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT                                          \
    void KCGenerateCamRays<typename C::Camera, T>(MRAY_GRID_CONSTANT const Span<RayCone>,           \
                                                  MRAY_GRID_CONSTANT const Span<RayGMem>,           \
                                                  MRAY_GRID_CONSTANT const Span<ImageCoordinate>,   \
                                                  MRAY_GRID_CONSTANT const Span<Float>,             \
                                                  MRAY_GRID_CONSTANT const Span<const uint32_t>,    \
                                                  MRAY_GRID_CONSTANT const Span<const RandomNumber>,\
                                                  MRAY_GRID_CONSTANT const C::Camera* const,        \
                                                  MRAY_GRID_CONSTANT const TransformKey,            \
                                                  MRAY_GRID_CONSTANT const typename T::DataSoA,     \
                                                  MRAY_GRID_CONSTANT const uint64_t,                \
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
