#pragma once

#include "RendererC.h"
#include "RayPartitioner.h"
#include "MediaTracker.h"

// Some common kernels
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateSurfaceWorkKeys(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                               MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                               MRAY_GRID_CONSTANT const RenderSurfaceWorkHasher workHasher);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateSurfaceWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                                       MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                       MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                                       MRAY_GRID_CONSTANT const RenderSurfaceWorkHasher workHasher);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateMediumWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                                      MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                      MRAY_GRID_CONSTANT const Span<const RayMediaListPack> dRayMLPacks,
                                      MRAY_GRID_CONSTANT const MediaTrackerView tracker,
                                      MRAY_GRID_CONSTANT const RenderMediumWorkHasher workHasher);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetBoundaryWorkKeys(MRAY_GRID_CONSTANT const Span<HitKeyPack> dWorkKey,
                           MRAY_GRID_CONSTANT const HitKeyPack boundaryWorkKey);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetBoundaryWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<HitKeyPack> dWorkKey,
                                   MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                   MRAY_GRID_CONSTANT const HitKeyPack boundaryWorkKey);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCCopyRaysIndirect(MRAY_GRID_CONSTANT const Span<RayGMem> dRaysOut,
                        MRAY_GRID_CONSTANT const Span<RayCone> dRayDiffOut,
                        MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                        MRAY_GRID_CONSTANT const Span<const RayGMem> dRaysIn,
                        MRAY_GRID_CONSTANT const Span<const RayCone> dRayDiffIn);

template<class Renderer, class WorkF, class LightWorkF, class CamWorkF>
void RendererBase::IssueSurfaceWorkKernelsToPartitions(const RenderSurfaceWorkHasher& workHasher,
                                                       const MultiPartitionOutput& p,
                                                       WorkF&& WF, LightWorkF&& LWF,
                                                       CamWorkF&&) const
{
    assert(p.isHostVisible);
    for(uint32_t i = 0; i < p.hPartitionCount[0]; i++)
    {
        uint32_t partitionStart = p.hdPartitionStartOffsets[i];
        uint32_t partitionSize = (p.hdPartitionStartOffsets[i + 1] -
                                  p.hdPartitionStartOffsets[i]);
        auto dLocalIndices = p.dPartitionIndices.subspan(partitionStart,
                                                         partitionSize);

        // Find the work
        // TODO: Although work count should be small,
        // doing a linear search here may not be performant.
        CommonKey key = workHasher.BisectBatchPortion(p.hdPartitionKeys[i]);
        auto wLoc = std::find_if(this->currentWorks.cbegin(),
                                 this->currentWorks.cend(),
                                 [key](const auto& workInfo)
        {
            return workInfo.workGroupId == key;
        });
        auto lightWLoc = std::find_if(this->currentLightWorks.cbegin(),
                                      this->currentLightWorks.cend(),
                                      [key](const auto& workInfo)
        {
            return workInfo.workGroupId == key;
        });
        if(wLoc != this->currentWorks.cend())
        {
            const auto& workPtr = UpcastRenderWork<Renderer>(wLoc->workPtr);
            WF(workPtr, dLocalIndices, partitionStart, partitionSize);
        }
        else if(lightWLoc != this->currentLightWorks.cend())
        {
            const auto& workPtr = UpcastRenderLightWork<Renderer>(lightWLoc->workPtr);
            LWF(workPtr, dLocalIndices, partitionStart, partitionSize);
        }
        else throw MRayError("[{}]: Unkown work id is found ({}).",
                             rendererName, key);
    }
}

template<class Renderer, class WorkF>
void RendererBase::IssueMediumWorkKernelsToPartitions(const RenderMediumWorkHasher& workHasher,
                                                      const MultiPartitionOutput& p,
                                                      WorkF&& WF) const
{
    assert(p.isHostVisible);
    for(uint32_t i = 0; i < p.hPartitionCount[0]; i++)
    {
        uint32_t partitionStart = p.hdPartitionStartOffsets[i];
        uint32_t partitionSize = (p.hdPartitionStartOffsets[i + 1] -
                                  p.hdPartitionStartOffsets[i]);
        auto dLocalIndices = p.dPartitionIndices.subspan(partitionStart,
                                                         partitionSize);

        // Find the work
        // TODO: Although work count should be small,
        // doing a linear search here may not be performant.
        CommonKey key = workHasher.BisectBatchPortion(p.hdPartitionKeys[i]);
        auto wLoc = std::find_if(this->currentMediumWorks.cbegin(),
                                 this->currentMediumWorks.cend(),
                                 [key](const auto& workInfo)
        {
            return workInfo.workGroupId == key;
        });
        if(wLoc != this->currentWorks.cend())
        {
            const auto& workPtr = UpcastRenderWork<Renderer>(wLoc->workPtr);
            WF(workPtr, dLocalIndices, partitionStart, partitionSize);
        }
        else throw MRayError("[{}]: Unkown work id is found ({}).",
                             rendererName, key);
    }
}