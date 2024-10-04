#pragma once

#include "RendererC.h"
#include "RayPartitioner.h"

// Some common kernels
MRAY_KERNEL
void KCGenerateWorkKeys(MRAY_GRID_CONSTANT const Span<CommonKey> dWorkKey,
                        MRAY_GRID_CONSTANT const Span<const HitKeyPack> dInputKeys,
                        MRAY_GRID_CONSTANT const RenderWorkHasher workHasher);

MRAY_KERNEL
void KCSetBoundaryWorkKeys(MRAY_GRID_CONSTANT const Span<HitKeyPack> dWorkKey,
                           MRAY_GRID_CONSTANT const HitKeyPack boundaryWorkKey);

MRAY_KERNEL
void KCSetBoundaryWorkKeysIndirect(MRAY_GRID_CONSTANT const Span<HitKeyPack> dWorkKey,
                                   MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                   MRAY_GRID_CONSTANT const HitKeyPack boundaryWorkKey);

template <class C>
template<class WorkF, class LightWorkF, class CamWorkF>
void RendererT<C>::IssueWorkKernelsToPartitions(const RenderWorkHasher& workHasher,
                                                const MultiPartitionOutput& p,
                                                WorkF&& WF, LightWorkF&& LWF,
                                                CamWorkF&&) const
{
    assert(partition.isHostVisible);
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
            const auto& workPtr = *wLoc->workPtr.get();
            WF(workPtr, dLocalIndices, partitionStart, partitionSize);
        }
        else if(lightWLoc != this->currentLightWorks.cend())
        {
            const auto& workPtr = *lightWLoc->workPtr.get();
            LWF(workPtr, dLocalIndices, partitionStart, partitionSize);
        }
        else throw MRayError("[{}]: Unkown work id is found ({}).",
                             C::TypeName(), key);
    }
}
