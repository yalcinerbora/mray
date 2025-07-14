#include "AcceleratorEmbree.h"
#include "TransformC.h"

#include <cassert>
#include <map>

#include "Core/System.h"
#include "Core/Expected.h"
#include "Core/Filesystem.h"

#include "Device/GPUAlgScan.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgGeneric.h"

std::string_view BaseAcceleratorEmbree::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Hardware"sv;
    return BaseAccelTypeName<Name>;
}

BaseAcceleratorEmbree::BaseAcceleratorEmbree(ThreadPool& tp, const GPUSystem& sys,
                                           const AccelGroupGenMap& genMap,
                                           const AccelWorkGenMap& workGenMap)
    : BaseAcceleratorT<BaseAcceleratorEmbree>(tp, sys, genMap, workGenMap)
{

}

AABB3 BaseAcceleratorEmbree::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    return AABB3::Covering();
}

void BaseAcceleratorEmbree::AllocateForTraversal(size_t)
{}

void BaseAcceleratorEmbree::CastRays(// Output
                                     Span<HitKeyPack> dHitIds,
                                     Span<MetaHit> dHitParams,
                                     // I-O
                                     Span<BackupRNGState> dRNGStates,
                                     Span<RayGMem> dRays,
                                     // Input
                                     Span<const RayIndex> dRayIndices,
                                     const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

}

void BaseAcceleratorEmbree::CastVisibilityRays(Bitspan<uint32_t> dIsVisibleBuffer,
                                               // I-O
                                               Span<BackupRNGState> dRNGStates,
                                               // Input
                                               Span<const RayGMem> dRays,
                                               Span<const RayIndex> dRayIndices,
                                               const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Visibilty Casting"sv);
    const auto _ = annotation.AnnotateScope();
}

void BaseAcceleratorEmbree::CastLocalRays(// Output
                                          Span<HitKeyPack> dHitIds,
                                          Span<MetaHit> dHitParams,
                                          // I-O
                                          Span<BackupRNGState> dRNGStates,
                                          Span<RayGMem> dRays,
                                          // Input
                                          Span<const RayIndex> dRayIndices,
                                          Span<const AcceleratorKey> dAccelKeys,
                                          CommonKey dAccelKeyBatchPortion,
                                          const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    const auto annotation = gpuSystem.CreateAnnotation("Local Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();


}

size_t BaseAcceleratorEmbree::GPUMemoryUsage() const
{
    return 0u;
}