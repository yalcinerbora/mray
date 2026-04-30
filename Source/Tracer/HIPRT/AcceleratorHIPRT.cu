#include "AcceleratorHIPRT.h"
#include "TransformC.h"

std::string_view BaseAcceleratorHIPRT::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Hardware"sv;
    return BaseAccelTypeName<Name>;
}

BaseAcceleratorHIPRT::BaseAcceleratorHIPRT(ThreadPool& tp, const GPUSystem& sys,
                                           const AccelGroupGenMap& genMap,
                                           const AccelWorkGenMap& workGenMap)
    : BaseAcceleratorT<BaseAcceleratorHIPRT>(tp, sys, genMap, workGenMap)
{}

AABB3 BaseAcceleratorHIPRT::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    return AABB3::Negative();
}

void BaseAcceleratorHIPRT::AllocateForTraversal(size_t)
{}

void BaseAcceleratorHIPRT::CastRays(// Output
                                    Span<VolumeIndex> dVolumeIndices,
                                    Span<HitKeyPack> dHitIds,
                                    Span<MetaHit> dHitParams,
                                    // I-O
                                    Span<BackupRNGState> dRNGStates,
                                    Span<RayGMem> dRays,
                                    // Input
                                    Span<const RayIndex> dRayIndices,
                                    //
                                    bool resolveMedia,
                                    const GPUQueue& queue)
{}

void BaseAcceleratorHIPRT::CastVisibilityRays(Bitspan<uint32_t> dIsVisibleBuffer,
                                              // I-O
                                              Span<BackupRNGState> dRNGStates,
                                              // Input
                                              Span<const RayGMem> dRays,
                                              Span<const RayIndex> dRayIndices,
                                              const GPUQueue& queue)
{}

void BaseAcceleratorHIPRT::CastLocalRays(// Output
                                         Span<VolumeIndex> dVolumeIndices,
                                         Span<HitKeyPack> dHitIds,
                                         Span<MetaHit> dHitParams,
                                         // I-O
                                         Span<BackupRNGState> dRNGStates,
                                         Span<RayGMem> dRays,
                                         // Input
                                         Span<const RayIndex> dRayIndices,
                                         Span<const AcceleratorKey> dAccelKeys,
                                         //
                                         CommonKey dAccelKeyBatchPortion,
                                         bool resolveMedia,
                                         const GPUQueue& queue)
{}

size_t BaseAcceleratorHIPRT::GPUMemoryUsage() const
{
    return 0;
}