#pragma once

#include "TracerTypes.h"
#include "Random.h"
#include "Bitspan.h"

#include "Core/Types.h"

#include "Device/GPUSystemForward.h"

// Accelerator can write volumes,
// surface key tuples and hit parameters (i.e., barycentric coords)
// and volume indices (which volume this surface splits)
//
// Some code may need only one or both:
//   - Path tracing *without* media will need surface hit keys only
//   - Volume renderer that traces shadow rays needs only volume index
//     (volume index holds if the surface is "passthrough" so we can recursively
//      shed radiance by using volume index only)
//   - Path tracer *with* media will require both when evaluating path rays
//
// We use this to prevent writing to certain buffers so that algorithms
// may repurpose that memory. (i.e, volume shadow ray casting will repurpose
// hit key buffers)
enum class AccelResultWriteMode : uint32_t
{
    HIT_KEY_AND_HIT_ONLY = 0b01,
    VOLUME_INDEX_ONLY    = 0b10,
    BOTH                 = 0b11
};

class AcceleratorWorkI
{
    public:
    virtual         ~AcceleratorWorkI() = default;

    virtual void    CastLocalRays(// Output
                                  Span<VolumeIndex> dVolumeIndices,
                                  Span<HitKeyPack> dHitKeys,
                                  Span<MetaHit> dHitParams,
                                  // I-O
                                  Span<BackupRNGState> dRNGStates,
                                  Span<RayGMem> dRays,
                                  // Input
                                  Span<const RayIndex> dRayIndices,
                                  Span<const CommonKey> dAccelIdPacks,
                                  // Constants
                                  AccelResultWriteMode,
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
    virtual void    GetCommonTransforms(Span<Matrix3x4> dTransforms,
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
