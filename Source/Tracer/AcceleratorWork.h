#include "AcceleratorC.h"
#include "TransformC.h"
#include "PrimitiveC.h"
#include "Random.h"
#include "TracerTypes.h"

template<PrimitiveGroupC PG>
using OptionalHitR = Optional<HitResultT<typename PG::Hit>>;

template<AccelGroupC AG, TransformGroupC TG>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCLocalRayCast(// Output
                           MRAY_GRID_CONSTANT const Span<HitKeyPack> dHitIds,
                           // I-O
                           MRAY_GRID_CONSTANT const Span<BackupRNGState> rngStates,
                           MRAY_GRID_CONSTANT const Span<RayGMem> dRays,
                           // Input
                           MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                           MRAY_GRID_CONSTANT const Span<const CommonKey> dAcceleratorKeys,
                           // Constant
                           MRAY_GRID_CONSTANT const typename TG::DataSoA tSoA,
                           MRAY_GRID_CONSTANT const typename AG::DataSoA aSoA,
                           MRAY_GRID_CONSTANT const typename AG::PrimitiveGroup::DataSoA pSoA)
{
    using PG = typename AG::PrimitiveGroup;
    using Accelerator = typename AG:: template Accelerator<TG>;
    KernelCallParams kp;

    uint32_t workCount = static_cast<uint32_t>(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < workCount; i += kp.TotalSize())
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
        if constexpr(PG::TransformLogic == LOCALLY_CONSTANT_TRANSFORM)
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
        if(!hit) continue;

        dHitIds[i] = HitKeyPack
        {
            .primKey = hit.value().primitiveKey,
            .lightOrMatKey = hit.value().lmKey,
            .transKey = acc.TransformKey(),
            .accelKey = aId
        };
        UpdateTMax(dRays, index, hit.value().t);
    }
};

class AcceleratorWorkI
{
    private:
    protected:
    virtual void CastLocalRays(// Output
                               Span<HitKeyPack> dHitKeys,
                               // I-O
                               Span<BackupRNGState> rngStates,
                               Span<RayGMem> dRays,
                               // Input
                               Span<const RayIndex> dRayIndices,
                               Span<const CommonKey> dAccelIdPacks,
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
                       Span<RayGMem> dRays,
                       // Input
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
                                            Span<RayGMem> dRays,
                                            // Input
                                            Span<const RayIndex> dRayIndices,
                                            Span<const CommonKey> dAcceleratorKeys,

                                            // Constants
                                            const GPUQueue& queue) const
{
    assert(dRays.size() == dRayIndices.size());
    assert(dRayIndices.size() == dAcceleratorKeys.size());
    assert(dAcceleratorKeys.size() == dHitIds.size());

    using namespace std::string_literals;
    queue.IssueSaturatingKernel<KCLocalRayCast<AG, TG>>
    (
        std::string(TypeName()) + "-CastLocalRays"s,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dRays.size())},
        //
        dHitIds,
        rngStates,
        dRays,
        dRayIndices,
        dAcceleratorKeys,
        transGroup.SoA(),
        accelGroup.SoA(),
        primGroup.SoA()
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
