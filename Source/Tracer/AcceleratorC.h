#pragma once

#include <concepts>

#include "TracerTypes.h"
#include "Core/Types.h"
#include "Device/GPUSystemForward.h"
#include "Random.h"

class IdentityTransformContext;

template<class HitType>
struct HitResultT
{
    MaterialId      materialId;
    PrimitiveId     primitiveId;
    HitType         hit;
    Float           t;
};

// Main accelerator leaf, holds primitive id and
// material id. Most of the time this should be enough
struct AcceleratorLeaf
{
    MaterialId  materialId;
    PrimitiveId primitiveId;
};

struct BaseAcceleratorLeaf
{
    TransformId     transformId;
    AcceleratorId   accelId;
};

template <class AccelType>
concept AccelC = requires(AccelType acc)
{
    typename AccelType::template PrimType<>;
    typename AccelType::PrimHit;
    typename AccelType::HitResult;
    typename AccelType::DataSoA;

    // Has this specific constructor
    requires requires(const IdentityTransformContext& tc,
                      const typename AccelType::DataSoA& soaData)
    {AccelType(tc, soaData, AcceleratorId{});};

    // Closest hit and any hit
    {acc.ClosestHit(Ray{}, Vector2f{})
    } -> std::same_as<Optional<typename AccelType::HitResult>>;

    {acc.FirstHit(Ray{}, Vector2f{})
    }-> std::same_as<Optional<typename AccelType::HitResult>>;
};

template <class AccelGroupType>
concept AccelGroupC = requires()
{
    true;
};

template<class AccelInstance>
concept AccelInstanceGroupC = requires(AccelInstance ag,
                                       GPUSystem gpuSystem)
{
    {ag.CastLocalRays(// Output
                      Span<HitIdPack>{},
                      Span<MetaHit>{},
                      // Input
                      Span<const RayGMem>{},
                      Span<const RayIndex>{},
                      Span<const AccelIdPack>{},
                      // Constants
                      gpuSystem)} -> std::same_as<void>;
};

template <class BaseAccel>
concept BaseAccelC = requires(BaseAccel ac,
                              GPUSystem gpuSystem)
{
    {ac.CastRays(// Output
                 Span<HitIdPack>{},
                 Span<MetaHit>{},
                 Span<SurfaceWorkKey>{},
                 // Input
                 Span<const RayGMem>{},
                 Span<const RayIndex>{},
                 // Constants
                 gpuSystem)} -> std::same_as<void>;

    {ac.CastShadowRays(// Output
                       Bitspan<uint32_t>{},
                       Bitspan<uint32_t>{},
                       // Input
                       Span<const RayIndex>{},
                       Span<const RayGMem>{},
                       // Constants
                       gpuSystem)} -> std::same_as<void>;
};

class AcceleratorGroupI
{
    public:
    virtual     ~AcceleratorGroupI() = default;

    virtual void Construct(AcceleratorId) = 0;
    virtual void Reconstruct(AcceleratorId) = 0;
};

class AcceleratorInstanceGroupI
{
    public:
    virtual         ~AcceleratorInstanceGroupI() = default;

    // Interface
    virtual void    CastLocalRays(// Output
                                  Span<HitIdPack> dHitIds,
                                  Span<MetaHit> dHitParams,
                                  // I-O
                                  Span<BackupRNGState> rngStates,
                                  // Input
                                  Span<const RayGMem> dRays,
                                  Span<const RayIndex> dRayIndices,
                                  Span<const AcceleratorIdPack> dAccelIdPacks,
                                  // Constants
                                  const GPUSystem& s) = 0;
};

class AcceleratorBaseI
{
    public:
    virtual     ~AcceleratorBaseI() = default;

    // Interface
    virtual void CastRays(// Output
                          Span<HitIdPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          Span<SurfaceWorkKey> dWorkKeys,
                          // Input
                          Span<const RayGMem> dRays,
                          Span<const RayIndex> dRayIndices,
                          // Constants
                          const GPUSystem& s) = 0;

    virtual void CastShadowRays(// Output
                                Bitspan<uint32_t> dIsVisibleBuffer,
                                Bitspan<uint32_t> dFoundMediumInterface,
                                // Input
                                Span<const RayIndex> dRayIndices,
                                Span<const RayGMem> dShadowRays,
                                // Constants
                                const GPUSystem& s) = 0;

    virtual void Construct() = 0;
    virtual void Reconstruct() = 0;
};


static_assert(AccelInstanceGroupC<AcceleratorInstanceGroupI>, "");

// Support Concepts
//template <class AccelType, class AccelGroupType, class PrimType>
//concept AccelWithTransformC = requires(PrimType tg)
//{
//    requires AccelC<AccelType, AccelGroupType>;
//};