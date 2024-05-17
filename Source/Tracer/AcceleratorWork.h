#include "AcceleratorC.h"
#include "TransformC.h"
#include "PrimitiveC.h"
#include "Random.h"

class AcceleratorGroupI;

class AcceleratorWorkI
{
    private:
    protected:
    virtual void CastLocalRays(// Output
                               Span<HitKeyPack> dHitKeys,
                               // I-O
                               Span<BackupRNGState> rngStates,
                               // Input
                               Span<const RayGMem> dRays,
                               Span<const RayIndex> dRayIndices,
                               Span<const AcceleratorKey> dAccelIdPacks,
                               // Constants
                               const GPUQueue& queue) const = 0;
};

template<AccelGroupC AcceleratorGroupType,
         TransformGroupC TransformGroupType>
class AcceleratorWork : public AcceleratorWorkI
{
    public:
    using TransformGroup    = TransformGroupType;
    using AcceleratorGroup  = AcceleratorGroupType;
    using PrimitiveGroup    = typename AcceleratorGroup::PrimitiveGroup;

    static std::string_view TypeName() const;

    private:
    const PrimitiveGroup&       primGroup;
    const AcceleratorGroup&     accelGroup;
    const TransformGroup&       transGroup;

    public:
    AcceleratorWork(const AcceleratorGroupI& ag,
                    const GenericGroupTransformT& tg);

    // Cast Local rays
    void CastLocalRays(// Output
                       Span<HitKeyPack> dHitKeys,
                       // I-O
                       Span<BackupRNGState> rngStates,
                       // Input
                       Span<const RayGMem> dRays,
                       Span<const RayIndex> dRayIndices,
                       Span<const CommonKey> dAcceleratorKeys,
                       // Constants
                       const GPUQueue& queue) override;

};


template<AccelGroupC AG, TransformGroupC TG>
AcceleratorWork<AG, TG>::AcceleratorInstanceGroup(const AcceleratorGroupI& ag,
                                                  const GenericGroupTransformT& tg)
    : accelGroup(static_cast<const AG&>(ag))
    , transGroup(static_cast<const TG&>(tgIn))
    , primGroup(static_cast<const PrimitiveGroup&>(*ag.PrimGroup()))
{

}

template<AccelGroupC AG, TransformGroupC TG>
void AcceleratorWork<AG, TG>::CastLocalRays(// Output
                                            Span<HitKeyPack> dHitIds,
                                            // I-O
                                            Span<BackupRNGState> rngStates,
                                            // Input
                                            Span<const RayGMem> dRays,
                                            Span<const RayIndex> dRayIndices,
                                            Span<const CommonKey> dAcceleratorKeys,

                                            // Constants
                                            const GPUQueue& queue)
{
    assert(dRays.size() == dRayIndices.size());
    assert(dRayIndices.size() == dAccelIdPacks.size());
    assert(dAccelIdPacks.size() == dHitParams.size());
    assert(dHitParams.size() == dHitIds.size());

    using PG            = typename AcceleratorGroup::PrimitiveGroup;
    using PrimSoA       = typename PrimitiveGroup::DataSoAConst;
    using AccelSoA      = typename AcceleratorGroup::DataSoAConst;
    using TransSoA      = typename TransformGroup::DataSoAConst;
    using Accelerator   = typename AcceleratorGroup:: template Accelerator<TG>;
    TransSoA    tSoA = tg.DataSoA();
    AccelSoA    aSoA = ag.DataSoA();
    PrimSoA     pSoA = pg.DataSoA();

    auto RayCastKernel = [=] MRAY_HYBRID(KernelCallParams kp)
    {
        uint32_t workCount = static_cast<uint32_t>(dRayIndices.size());
        for(uint32_t i = kp.GlobalId(); i < workCount; i+= kp.TotalSize())
        {
            RayIndex index = dRayIndices[i];
            auto [ray, tMM] = RayFromGMem(dRays, index);

            BackupRNG rng(rngStates[index]);

            // Do work depending on the prim transorm logic
            using enum PrimTransformType;
            if constexpr(PG::TransformType == LOCALLY_CONSTANT_TRANSFORM)
            {
                // Transform is local
                // we can transform the ray and use it on iterations
                // Compile-time find the transform generator function and return type
                static constexpr
                auto TContextGen = AcquireTransformContextGenerator<PG, TG>();
                constexpr auto TGenFunc = decltype(TContextGen)::Function;
                // Define the types
                // First, this kernel uses a transform context
                // that this primitive group provides to generate the prim
                using TContextType = typename decltype(TContextGen)::ReturnType;
                // The actual context
                TContextType transformContext = TGenFunc(transformSoA, primitiveSoA,
                                                         transformKey, PrimitiveKey(0));

                ray = transformContext.InvApply(ray);
            }

            // Get ids
            auto [tId, aId] = dAccelIdPacks[index];

            // Construct the accelerator view
            Accelerator acc(tSoA, pSoA, aSoA, tId, aId);
            OptionalHitR<PrimitiveGroup> hit = acc.ClosestHit(ray, tMM, rng);

            if(hit)
            {
                dHitIds[i] = HitIdPack
                {
                    .primId     = hit.primitiveKey,
                    .matId      = hit.materialKey,
                    .transId    = tId,
                    .accelId    = aId
                };
                dHitParams[i] = hit.hit;
                UpdateTMax(gRays, index, hit.t);
            }
        }
    };

    queue.IssueLambda
    (
        TypeName() + "-CastLocalRays"sv,
        KernelIssueParams{.workSize = static_cast<uint32_t>(dRays.size())},
        std::move(RayCastKernel)
    );
}

template<AccelGroupC AG, TransformGroupC TG>
std::string_view AcceleratorInstanceGroup<AG, TG>::TypeName()
{
    static std::string name = AG::TypeName() + TG::TypeName();
    return name;
}
