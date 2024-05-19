#include "AcceleratorC.h"
#include "TransformC.h"
#include "PrimitiveC.h"
#include "Random.h"
#include "TracerTypes.h"

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
                               Span<const CommonKey> dAccelIdPacks,
                               // Constants
                               const GPUQueue& queue) const = 0;
};

template<PrimitiveGroupC PG>
using OptionalHitR = Optional<HitResultT<typename PG::Hit>>;

template<AccelGroupC AcceleratorGroupType,
         TransformGroupC TransformGroupType>
class AcceleratorWork : public AcceleratorWorkI
{
    public:
    using TransformGroup    = TransformGroupType;
    using AcceleratorGroup  = AcceleratorGroupType;
    using PrimitiveGroup    = typename AcceleratorGroup::PrimitiveGroup;

    static std::string_view TypeName();

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
                       const GPUQueue& queue) const override;

};


template<AccelGroupC AG, TransformGroupC TG>
AcceleratorWork<AG, TG>::AcceleratorWork(const AcceleratorGroupI& ag,
                                         const GenericGroupTransformT& tg)
    : accelGroup(static_cast<const AG&>(ag))
    , transGroup(static_cast<const TG&>(tg))
    , primGroup(static_cast<const PrimitiveGroup&>(ag.PrimGroup()))
{}

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
                                            const GPUQueue& queue) const
{
    assert(dRays.size() == dRayIndices.size());
    assert(dRayIndices.size() == dAcceleratorKeys.size());
    assert(dAcceleratorKeys.size() == dHitIds.size());

    using PG            = typename AcceleratorGroup::PrimitiveGroup;
    using PrimSoA       = typename PrimitiveGroup::DataSoA;
    using AccelSoA      = typename AcceleratorGroup::DataSoA;
    using TransSoA      = typename TransformGroup::DataSoA;
    using Accelerator   = typename AcceleratorGroup:: template Accelerator<TG>;
    TransSoA    tSoA = transGroup.DataSoA();
    AccelSoA    aSoA = accelGroup.DataSoA();
    PrimSoA     pSoA = primGroup.DataSoA();

    auto RayCastKernel = [=] MRAY_HYBRID(KernelCallParams kp)
    {
        uint32_t workCount = static_cast<uint32_t>(dRayIndices.size());
        for(uint32_t i = kp.GlobalId(); i < workCount; i+= kp.TotalSize())
        {
            RayIndex index = dRayIndices[i];
            auto [ray, tMM] = RayFromGMem(dRays, index);

            BackupRNG rng(rngStates[index]);

            // Get ids
            AcceleratorKey aId(dAcceleratorKeys[index]);
            // Construct the accelerator view
            Accelerator acc(tSoA, pSoA, aSoA, aId);

            // Do work depending on the prim transorm logic
            using enum PrimTransformType;
            if(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
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
                // Prim id does mean nothing, so set it to zero and call
                TContextType transformContext = TGenFunc(tSoA, pSoA, acc.TransformKey(),
                                                         PrimitiveKey(0));

                ray = transformContext.InvApply(ray);
            }

            // Actual ray cast!
            OptionalHitR<PG> hit = acc.ClosestHit(rng, ray, tMM);

            if(hit)
            {
                dHitIds[i] = HitKeyPack
                {
                    .primKey        = hit.value().primitiveKey,
                    .lightOrMatKey  = hit.value().materialKey,
                    .transKey       = acc.TransformKey(),
                    .accelKey       = aId
                };
                UpdateTMax(dRays, index, hit.t);
            }
        }
    };

    using namespace std::string_literals;
    queue.IssueLambda
    (
        std::string(TypeName()) + "-CastLocalRays"s,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dRays.size())},
        std::move(RayCastKernel)
    );
}

template<AccelGroupC AG, TransformGroupC TG>
std::string_view AcceleratorWork<AG, TG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const std::string Name = AccelWorkTypeName(AG::TypeName(),
                                                      TG::TypeName());
    return Name;
}
