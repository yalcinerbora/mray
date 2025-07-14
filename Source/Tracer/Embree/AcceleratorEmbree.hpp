#pragma once

#include "AcceleratorEmbree.h"

template<PrimitiveGroupC PG>
std::string_view AcceleratorGroupEmbree<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = AccelGroupTypeName(BaseAcceleratorEmbree::TypeName(),
                                                PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupEmbree<PG>::AcceleratorGroupEmbree(uint32_t accelGroupId,
                                                   ThreadPool& tp,
                                                   const GPUSystem& sys,
                                                   const GenericGroupPrimitiveT& pg,
                                                   const AccelWorkGenMap& wMap)
    : Base(accelGroupId, tp, sys, pg, wMap)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::PreConstruct(const BaseAcceleratorI* a)
{
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::Construct(AccelGroupConstructParams p,
                                           const GPUQueue& queue)
{

}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::WriteInstanceKeysAndAABBs(Span<AABB3>,
                                                           Span<AcceleratorKey>,
                                                           const GPUQueue&) const
{
    throw MRayError("For Embree, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::CastLocalRays(// Output
                                               Span<HitKeyPack>,
                                               Span<MetaHit>,
                                               // I-O
                                               Span<BackupRNGState>,
                                               Span<RayGMem>,
                                               // Input
                                               Span<const RayIndex>,
                                               Span<const CommonKey>,
                                               // Constants
                                               CommonKey,
                                               const GPUQueue&)
{
    throw MRayError("For Embree, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::CastVisibilityRays(// Output
                                                    Bitspan<uint32_t>,
                                                    // I-O
                                                    Span<BackupRNGState>,
                                                    // Input
                                                    Span<const RayGMem>,
                                                    Span<const RayIndex>,
                                                    Span<const CommonKey>,
                                                    // Constants
                                                    CommonKey,
                                                    const GPUQueue&)
{
    throw MRayError("For OptiX, this function should not be called");
}

template<PrimitiveGroupC PG>
typename AcceleratorGroupEmbree<PG>::DataSoA
AcceleratorGroupEmbree<PG>::SoA() const
{
    return EmptyType{};
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupEmbree<PG>::GPUMemoryUsage() const
{
    return 0;
}