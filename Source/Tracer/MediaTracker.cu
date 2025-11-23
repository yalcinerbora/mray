#include "MediaTracker.h"

#include "Device/GPUSystem.hpp"

#include "Core/TracerConstants.h"
#include "Core/TracerI.h"

uint32_t MediaTracker::FindVolumeIndex(VolumeId vId) const
{
    auto loc = std::lower_bound(globalVolumeList.cbegin(),
                                globalVolumeList.cend(),
                                Pair(vId, VolumeKeyPack{}),
    [](const auto& a, const auto& b)
    {
        return a.first < b.first;
    });
    if(loc != globalVolumeList.cend() &&
       loc->first == vId)
    {
        return uint32_t(std::distance(globalVolumeList.cbegin(), loc));
    }
    else return UINT32_MAX;
}

// Constructors & Destructor
MediaTracker::MediaTracker(const VolumeList& globalVolumeList,
                           uint32_t maximumEntryCount,
                           const GPUSystem& gpuSystem)
    : gpuSystem(gpuSystem)
    , globalVolumeList(globalVolumeList)
    , mem(gpuSystem.AllGPUs(), 2_MiB, 4_MiB, true)
{
    assert(std::is_sorted(globalVolumeList.cbegin(), globalVolumeList.cend(),
                          [](const auto& a, const auto& b)
    {
        return a.first < b.first;
    }));

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    // Assume %50 load factor
    maximumEntryCount *= 2;
    MemAlloc::AllocateMultiData(Tie(dGlobalVolumeList,
                                    dLocksHT, dMediaListHT),
                                mem,
                                {globalVolumeList.size(),
                                 maximumEntryCount, maximumEntryCount});

    // Strip VolumeId from volume list for GPU
    std::vector<VolumeKeyPack> kp;
    kp.reserve(globalVolumeList.size());
    for(const auto pair : globalVolumeList)
        kp.push_back(pair.second);

    // Copy
    queue.MemcpyAsync(dGlobalVolumeList, Span<const VolumeKeyPack>(kp));
    queue.MemsetAsync(dLocksHT, 0x00);
    queue.MemsetAsync(dMediaListHT, 0xFF);
    assert(MediaTrackerView::EMPTY_VAL == UINT32_MAX);

    // TODO: This may be redundant
    queue.Barrier().Wait();
}

void MediaTracker::SetStartingVolumeIndirect(// Output
                                             Span<RayMediaListPack> packs,
                                             // Input
                                             Span<const RayIndex> dRayIndices,
                                             // Constants
                                             const BoundaryVolumeList& startVolumeList,
                                             const GPUQueue& queue)
{
    //
}

void MediaTracker::PrimeHashTable(const std::vector<const SurfaceVolumeList*>& hSurfaceVolumeList,
                                  const std::vector<const BoundaryVolumeList*>& hBoundaryVolumeList,
                                  VolumeId boundaryVolume,
                                  const GPUQueue& queue)
{
    using namespace TracerConstants;
    size_t conservativeSize = hSurfaceVolumeList.size() * 2 * MaxPrimBatchPerSurface;
    conservativeSize += hBoundaryVolumeList.size() * 2;

    // TODO: These should be on GPU
    std::vector<MediaList> mediaLists; mediaLists.reserve(conservativeSize);
    for(const SurfaceVolumeList* ptr : hSurfaceVolumeList)
    for(VolumeId vId : *ptr)
    {
        if(vId == InvalidVolume) continue;

        // We can only assume these are bounded by boundary volume
        // So only adding these to the HT
        // Others complex interaction will be handled automatically
        std::array<uint32_t, MAX_NESTED_MEDIA> unpackedList;
        unpackedList.fill(MediaTrackerView::EMPTY_VAL);
        unpackedList[0] = FindVolumeIndex(boundaryVolume);
        unpackedList[1] = FindVolumeIndex(vId);

        MediaList l; l.Pack(unpackedList);
        mediaLists.push_back(l);
    }

    // TODO: We are calculating all 2^8 (256) combinations
    // and adding these to the list.
    // This may be a performance issue when there are too many
    // boundary lights/cameras that are nested inside
    // of volumes (Which is very unlikely).
    // Then we need to put this either on the GPU
    // or get the CPU Thread pool and calculate it.
    for(const BoundaryVolumeList* ptr : hBoundaryVolumeList)
    {
        const BoundaryVolumeList& bList = *ptr;
        // Add all of the combinations of this list
        uint32_t totalCombinations = 1u << MAX_NESTED_MEDIA;
        std::array<uint32_t, MAX_NESTED_MEDIA> unpackedList;
        StaticVector<uint32_t, MAX_NESTED_MEDIA> volIndices;

        for(VolumeId v : bList)
            volIndices.push_back(FindVolumeIndex(v));

        for(uint32_t i = 1; i < totalCombinations; i++)
        {
            unpackedList.fill(MediaTrackerView::EMPTY_VAL);
            uint32_t counter = 0;
            for(uint32_t j = 0; j < MAX_NESTED_MEDIA; i++)
            {
                if(Bit::FetchSubPortion(i, {j, j + 1}) == 0) continue;
                unpackedList[counter++] = volIndices[j];
            }
            MediaList l; l.Pack(unpackedList);
            mediaLists.push_back(l);
        }
    }

    // Send to GPU and let GPU do the hash table stuff.
    //.............
    DeviceLocalMemory localMem(*queue.Device());
    Span<MediaList> dLists;
    MemAlloc::AllocateMultiData(Tie(dLists), localMem,
                                {mediaLists.size()});

    queue.MemcpyAsync(dLists, Span<const MediaList>(mediaLists));



}