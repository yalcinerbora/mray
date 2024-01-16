#pragma once

#include "Core/Types.h"
#include "Device/GPUSystem.h"
#include "TracerTypes.h"

// Double buffer of rays
// Partition

class RayCaster
{
    protected:
    DeviceMemory partitionCache;

    public:
    virtual ~RayCaster() = default;

    // Cast Rays and Find HitIds, and WorkKeys.
    // (the last one will be used to partition)
    virtual void CastRays(// Output
                          Span<HitIdPack>& dHitIds,     // Per-ray hit indices, which type of transform/primitive/material combo that this ray is hit
                          Span<MetaHit>& dHitParams,    // Per-ray hit parameters, where specifically this ray hit on that primitive (i.e. barycrentic coords for triangles)
                          // I-O
                          Span<WorkKey> dWorkKeys,     // Per-ray partition key (updated only when a hit occurs, remains the same if ray is missed)
                          Span<RayIndex> dRayIndices,  // Index buffer of the rays (may be partitioned else it is iota)
                          Span<RayGMem> dRays,         // Rays that will be casted to the entire scene (tMax will be updated)
                          // Constants
                          const GPUSystem& s) = 0;

    // Shadow ray?
    virtual void CastShadowRays(// Output
                                //Span<HitIdPack>& dHitIds,     // Per-ray hit indices, which type of transform/primitive/material combo that this ray is hit
                                //Span<MetaHit>& dHitParams,    // Per-ray hit parameters, where specifically this ray hit on that primitive (i.e. barycrentic coords for triangles)
                                // I-O
                                Span<WorkKey> dWorkKeys,     // Per-ray partition key (updated only when a hit occurs, remains the same if ray is missed)
                                Span<RayIndex> dRayIndices,  // Index buffer of the rays (may be partitioned else it is iota)
                                Span<RayGMem> dRays,         // Rays that will be casted to the entire scene (tMax will be updated)
                                // Constants
                                const GPUSystem& s) = 0;

    // Local ray casting (for BSSRDFs)
    virtual void CastLocalRays(// Output
                               Span<WorkKey> dWorkKeys,         // Per-ray partition key (updated only when a hit occurs, remains the same if ray is missed)
                               Span<HitIdPack>& dHitIds,        // Per-ray output, where the ray is hit
                               Span<MetaHit>& dHitParams,       //
                               // I-O
                               Span<RayIndex> dRayIndices,      // Index buffer of the rays (may be partitioned else it is iota)
                               Span<RayGMem> dRays,             // Rays that will be casted (via accelerator)
                               // Input
                               Span<const AccelKey> dRayKeys,   // Per-ray accelerator key, ray will be casted only this accelerator
                               // Constants
                               const GPUSystem& s) = 0;

    // Given per-ray custom data, partition wrt. to this data
};

