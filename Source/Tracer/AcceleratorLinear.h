#pragma once

#include "Core/Types.h"
#include "Device/GPUSystemForward.h"
#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "AcceleratorC.h"
#include "PrimitiveC.h"
#include "TransformC.h"
#include "Random.h"

namespace LinearAccelDetail
{
    using AlphaMap = ParamVaryingData<2, Float>;

    // SoA data of triangle group
    struct LinearAcceleratorSoA
    {
        // Per accelerator stuff
        Bitspan<uint32_t>             dCullFace;
        Span<Optional<AlphaMap>>      dAlphaMaps;
        Span<Span<AcceleratorLeaf>>   dLeafs;
    };

    template<PrimitiveGroupC PrimGroup, TransformGroupC TransGroupType>
    class AcceleratorLinear
    {
        using PrimHit       = typename PrimGroup::Hit;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = LinearAcceleratorSoA;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroupType::DataSoA;

        private:
        // Accel Related
        bool                        cullFace;
        Span<const AcceleratorLeaf> leafs;
        const Optional<AlphaMap>&   alphaMap;
        // Primitive Related
        TransformId                 transformId;
        const TransDataSoA&         transformSoA;
        const PrimDataSoA&          primitiveSoA;

        MRAY_HYBRID
        Optional<HitResult>     IntersectionCheck(const Ray& ray,
                                                  const Vector2& tMinMax,
                                                  Float xi,
                                                  const AcceleratorLeaf& l) const;
        public:
        MRAY_HYBRID             AcceleratorLinear(const TransDataSoA& tSoA,
                                                  const PrimDataSoA& pSoA,
                                                  const DataSoA& dataSoA,
                                                  AcceleratorId aId,
                                                  TransformId tId);

        MRAY_HYBRID
        Optional<HitResult>     ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const;
        MRAY_HYBRID
        Optional<HitResult>     FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const;
    };
}

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupLinear final : public AcceleratorGroupI
{
    public:
    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = LinearAccelDetail::LinearAcceleratorSoA;
    using PrimSoA           = PrimitiveGroupType::DataSoAConst;


    template<class TG>
    using Accelerator = LinearAccelDetail::AcceleratorLinear<PrimitiveGroup, TG>;

    private:
    //const PrimitiveGroup&   pg;
    DeviceMemory        mem;
    DataSoA             data;
};

class AcceleratorBaseLinear final : public AcceleratorBaseI
{

    // Cast Rays and Find HitIds, and WorkKeys.
    // (the last one will be used to partition)
    void CastRays(// Output
                  Span<HitIdPack> dHitIds,
                  Span<MetaHit> dHitParams,
                  Span<SurfaceWorkKey> dWorkKeys,
                  // Input
                  Span<const RayGMem> dRays,
                  Span<const RayIndex> dRayIndices,
                  // Constants
                  const GPUSystem& s) override;

    void CastShadowRays(// Output
                        Bitspan<uint32_t> dIsVisibleBuffer,
                        Bitspan<uint32_t> dFoundMediumInterface,
                        // Input
                        Span<const RayIndex> dRayIndices,
                        Span<const RayGMem> dShadowRays,
                        // Constants
                        const GPUSystem& s) override;

    // Each leaf has a surface
};




#include "AcceleratorLinear.hpp"