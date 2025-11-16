#pragma once

#include "TracerTypes.h"
#include "Random.h"
#include "Bitspan.h"

#include "Core/Types.h"

#include "Device/GPUSystemForward.h"

class AcceleratorWorkI
{
    public:
    virtual         ~AcceleratorWorkI() = default;

    virtual void    CastLocalRays(// Output
                                  Span<InterfaceIndex> dInterfaceIndices,
                                  Span<HitKeyPack> dHitKeys,
                                  Span<MetaHit> dHitParams,
                                  // I-O
                                  Span<BackupRNGState> dRNGStates,
                                  Span<RayGMem> dRays,
                                  // Input
                                  Span<const RayIndex> dRayIndices,
                                  Span<const CommonKey> dAccelIdPacks,
                                  // Constants
                                  bool writeInterfaceIndex,
                                  const GPUQueue& queue) const = 0;

    virtual void    CastVisibilityRays(// Output
                                       Bitspan<uint32_t> dIsVisibleBuffer,
                                       // I-O
                                       Span<BackupRNGState> dRNGStates,
                                       // Input
                                       Span<const RayGMem> dRays,
                                       Span<const RayIndex> dRayIndices,
                                       Span<const CommonKey> dAcceleratorKeys,
                                       // Constants
                                       const GPUQueue& queue) const = 0;

    virtual void    GeneratePrimitiveCenters(Span<Vector3> dAllPrimCenters,
                                             Span<const uint32_t> dLeafSegmentRanges,
                                             Span<const PrimitiveKey> dAllLeafs,
                                             Span<const TransformKey> dTransformKeys,
                                             const GPUQueue& queue) const = 0;
    virtual void    GeneratePrimitiveAABBs(Span<AABB3> dAllLeafAABBs,
                                           Span<const uint32_t> dLeafSegmentRanges,
                                           Span<const PrimitiveKey> dAllLeafs,
                                           Span<const TransformKey> dTransformKeys,
                                           const GPUQueue& queue) const = 0;
    // Transform related
    virtual void    GetCommonTransforms(Span<Matrix4x4> dTransforms,
                                        Span<const TransformKey> dTransformKeys,
                                        const GPUQueue& queue) const = 0;
    virtual void    TransformLocallyConstantAABBs(// Output
                                                  Span<AABB3> dInstanceAABBs,
                                                  // Input
                                                  Span<const AABB3> dConcreteAABBs,
                                                  Span<const uint32_t> dConcreteIndicesOfInstances,
                                                  Span<const TransformKey> dInstanceTransformKeys,
                                                  // Constants
                                                  const GPUQueue& queue) const = 0;
    virtual size_t  TransformSoAByteSize() const = 0;
    virtual void    CopyTransformSoA(Span<Byte>, const GPUQueue& queue) const = 0;

    virtual std::string_view TransformName() const = 0;
};
