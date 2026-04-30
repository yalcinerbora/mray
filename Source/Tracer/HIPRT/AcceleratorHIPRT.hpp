#pragma once

template<PrimitiveGroupC PG>
std::string_view AcceleratorGroupHIPRT<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = AccelGroupTypeName(BaseAcceleratorHIPRT::TypeName(),
                                                PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupHIPRT<PG>::AcceleratorGroupHIPRT(uint32_t accelGroupId,
                                                 ThreadPool& tp,
                                                 const GPUSystem& sys,
                                                 const GenericGroupPrimitiveT& pg,
                                                 const AccelWorkGenMap& wMap)
    : Base(accelGroupId, tp, sys, pg, wMap)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupHIPRT<PG>::PreConstruct(const BaseAcceleratorI* a)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupHIPRT<PG>::Construct(AccelGroupConstructParams p,
                                          const GPUQueue& queue)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupHIPRT<PG>::WriteInstanceKeysAndAABBs(Span<AABB3>,
                                                          Span<AcceleratorKey>,
                                                          const GPUQueue&) const
{
    throw MRayError("For HIPRT, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupHIPRT<PG>::OffsetAccelKeyInRecords()
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupHIPRT<PG>::CastLocalRays(// Output
                                              Span<VolumeIndex>,
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
                                              bool,
                                              const GPUQueue&)
{
    throw MRayError("For OptiX, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupHIPRT<PG>::CastVisibilityRays(// Output
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
typename AcceleratorGroupHIPRT<PG>::DataSoA
AcceleratorGroupHIPRT<PG>::SoA() const
{
    return EmptyType{};
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupHIPRT<PG>::GPUMemoryUsage() const
{
    return 0;
}