#pragma once

#include <concepts>

#include "TracerTypes.h"

#include "Core/Types.h"
#include "Core/TypeGenFunction.h"

#include "Device/GPUSystemForward.h"

#include "Random.h"

#include "PrimitiveC.h"
#include "LightC.h"

namespace BS { class thread_pool; }

class IdentityTransformContext;

template<class HitType>
struct HitResultT
{
    MaterialKey     materialKey;
    PrimitiveKey    primitiveKey;
    HitType         hit;
    Float           t;
};

// Main accelerator leaf, holds primitive id and
// material id. Most of the time this should be enough
struct AcceleratorLeaf
{
    PrimitiveKey primitiveKey;
    // TODO: Materials are comparably small wrt. primitives
    // so holding a 32-bit value for each triangle is a waste of space.
    // Reason and change this maybe?
    MaterialKey  materialKey;
};

struct BaseAcceleratorLeaf
{
    TransformKey     transformKey;
    AcceleratorKey   accelKey;
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
    {AccelType(tc, soaData, AcceleratorKey{});};

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
                      Span<const AcceleratorIdPack>{},
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

namespace TracerLimits
{
    static constexpr size_t MaxPrimBatchPerSurface = 8;
}


using PrimBatchListToAccelMapping = std::unordered_map<PrimBatchList, uint32_t>;
using LocalSurfaceToAccelMapping = std::unordered_map<uint32_t, uint32_t>;
//

using AlphaMap = TextureView<2, Float>;

struct BaseAccelConstructParams
{
    using SurfPair = Pair<SurfaceId, SurfaceParams>;
    using LightSurfPair = Pair<LightSurfaceId, LightSurfaceParams>;

    const std::map<PrimGroupId, PrimGroupPtr>&          primGroups;
    const std::map<LightGroupId, LightGroupPtr>&        lightGroups;
    const std::map<TransGroupId, TransformGroupPtr>&    transformGroups;
    const std::vector<SurfPair>&        mSurfList;
    const std::vector<LightSurfPair>&   lSurfList;
};

struct AccelGroupConstructParams
{
    using SurfPair              = typename BaseAccelConstructParams::SurfPair;
    using LightSurfPair         = typename BaseAccelConstructParams::LightSurfPair;
    using TGroupedSurfaces      = std::vector<Pair<TransGroupId, Span<SurfPair>>>;
    using TGroupedLightSurfaces = std::vector<Pair<TransGroupId, Span<LightSurfPair>>>;

    const std::map<TransGroupId, TransformGroupPtr>* transformGroups;
    const GenericGroupPrimitiveT*   primGroup;
    TGroupedSurfaces                tGroupSurfs;
    TGroupedLightSurfaces           tGroupLightSurfs;
};

class AcceleratorGroupI
{
    public:
    virtual         ~AcceleratorGroupI() = default;

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

    virtual void Construct(AccelGroupConstructParams) = 0;
};

using AccelGroupGenerator = GeneratorFuncType<AcceleratorGroupI, uint32_t,
                                              BS::thread_pool&, GPUSystem&>;

class BaseAcceleratorI
{
    public:
    virtual         ~BaseAcceleratorI() = default;

    // Interface
    virtual void    CastRays(// Output
                             Span<HitIdPack> dHitIds,
                             Span<MetaHit> dHitParams,
                             Span<SurfaceWorkKey> dWorkKeys,
                             // Input
                             Span<const RayGMem> dRays,
                             Span<const RayIndex> dRayIndices,
                             // Constants
                             const GPUSystem& s) = 0;

    virtual void    CastShadowRays(// Output
                                   Bitspan<uint32_t> dIsVisibleBuffer,
                                   Bitspan<uint32_t> dFoundMediumInterface,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   Span<const RayGMem> dShadowRays,
                                   // Constants
                                   const GPUSystem& s) = 0;

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

    // Construction
    virtual void    Construct(BaseAccelConstructParams p) = 0;
};

using AccelGroupPtr = std::unique_ptr<AcceleratorGroupI>;
using AcceleratorPtr = std::unique_ptr<BaseAcceleratorI>;

template <class Child>
class BaseAcceleratorT : public BaseAcceleratorI
{
    private:
    protected:
    BS::thread_pool&    threadPool;
    GPUSystem&          gpuSystem;
    uint32_t            idCounter = 0;

    //
    std::map<std::string_view, AccelGroupGenerator> accelGenerators;
    std::map<PrimGroupId, AccelGroupPtr>            generatedAccels;

    public:
                        BaseAcceleratorT(BS::thread_pool&, GPUSystem&,
                                         std::map<std::string_view, AccelGroupGenerator>&&);

};

template <class C>
BaseAcceleratorT<C>::BaseAcceleratorT(BS::thread_pool& tp, GPUSystem& system,
                                      std::map<std::string_view, AccelGroupGenerator>&& aGen)
    : threadPool(tp)
    , gpuSystem(system)
    , accelGenerators(aGen)
{}