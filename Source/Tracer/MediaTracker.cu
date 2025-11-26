#include "MediaTracker.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUWarp.h"

#include "Core/TracerConstants.h"
#include "Core/TracerI.h"

void SortByPrio(std::array<uint32_t, MAX_NESTED_MEDIA>& ml,
                std::array<int32_t, MAX_NESTED_MEDIA>& priorities,
                uint32_t indexCount)
{
    auto IsLess = [&](uint32_t j, uint32_t index, int32_t prio) -> bool
    {
        bool less = priorities[j] < prio;
        bool eqLess = (priorities[j] == prio);
        eqLess &= (ml[j] < index);
        return less || eqLess;
    };

    // Insertion Sort
    for(uint32_t i = 1; i < indexCount; i++)
    {
        int32_t prio = priorities[i];
        uint32_t index = ml[i];

        int32_t j = i - 1;
        for(; j >= 0 && IsLess(j, index, prio); j--)
        {
            std::swap(priorities[j + 1], priorities[j]);
            std::swap(ml[j + 1], ml[j]);
        }
        // Insert to the slot
        ml[j + 1] = index;
        priorities[j + 1] = prio;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCPrimeHashTable(MRAY_GRID_CONSTANT const MediaTrackerView tracker,
                      // Input
                      MRAY_GRID_CONSTANT const Span<const MediaList> dLists)
{
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < dLists.size(); i += kp.TotalSize())
        tracker.TryInsertAtomic(dLists[i]);
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetStartingVolumeIndirect(// Output
                                 MRAY_GRID_CONSTANT const Span<RayMediaListPack> dPacks,
                                 // Input
                                 MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                                 // Constants
                                 MRAY_GRID_CONSTANT const MediaList startVolumeList,
                                 MRAY_GRID_CONSTANT const MediaTrackerView tracker)
{
    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayIndex rIndex = dRayIndices[i];
        // TODO: Maybe implement a search?
        // Also we are searching for all threads
        // kinda wasteful (all will reach on to the same mem address so
        // it may not have large perf saving probably)
        auto [index, isInserted] = tracker.TryInsertAtomic(startVolumeList);
        assert(isInserted == false);

        RayMediaListPack p;
        p.SetCurMediaIndex(0);
        p.SetNextMediaIndex(0);
        p.SetOuterIndex(index);
        p.SetEntering(false);
        p.SetPassthrough(false);
        dPacks[rIndex] = p;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCAddNewVolumeToRayListIndirect(// I-O
                                     MRAY_GRID_CONSTANT const Span<RayMediaListPack> dPacks,
                                     MRAY_GRID_CONSTANT const MediaTrackerView tracker,
                                     //
                                     MRAY_GRID_CONSTANT const Span<const VolumeIndex> dNewVolumeIndices,
                                     MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayIndex rIndex = dRayIndices[i];
        RayMediaListPack p = dPacks[rIndex];
        VolumeIndex vI = dNewVolumeIndices[rIndex];
        tracker.InsertNewVolume(p, vI);
        dPacks[rIndex] = p;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCResolveCurVolumesOfRaysIndirect(// I-O
                                       MRAY_GRID_CONSTANT const Span<RayMediaListPack> dPacks,
                                       MRAY_GRID_CONSTANT const MediaTrackerView tracker,
                                       //
                                       MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices)
{
    KernelCallParams kp;
    uint32_t rayCount = uint32_t(dRayIndices.size());
    for(uint32_t i = kp.GlobalId(); i < rayCount; i += kp.TotalSize())
    {
        RayIndex rIndex = dRayIndices[i];
        RayMediaListPack p = dPacks[rIndex];
        tracker.ResolveCurVolume(p);
        dPacks[rIndex] = p;
    }
}

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
    maximumEntryCount = Math::NextPowerOfTwo(maximumEntryCount);
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
    // TODO: This may be redundant
    queue.Barrier().Wait();
}

void MediaTracker::SetStartingVolumeIndirect(// Output
                                             Span<RayMediaListPack> dPacks,
                                             // Input
                                             Span<const RayIndex> dRayIndices,
                                             // Constants
                                             const BoundaryVolumeList& startVolumeList,
                                             const GPUQueue& queue) const
{
    //
    std::array<uint32_t, MAX_NESTED_MEDIA> unpackedList;
    std::array<int32_t, MAX_NESTED_MEDIA> priorities;
    unpackedList.fill(MediaList::INVALID_VAL);

    uint32_t i = 0;
    for(VolumeId v : startVolumeList)
    {
        unpackedList[i] = FindVolumeIndex(v);
        priorities[i] = globalVolumeList[unpackedList[i]].second.priority;
        i++;
    }
    SortByPrio(unpackedList, priorities, i);
    MediaList l; l.PackAll(unpackedList);

    queue.IssueWorkKernel<KCSetStartingVolumeIndirect>
    (
        "KCSetStartingVolumeIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dRayIndices.size())},
        //
        dPacks,
        ToConstSpan(dRayIndices),
        l,
        View()
    );
}

void MediaTracker::PrimeHashTable(const std::vector<const SurfaceVolumeList*>& hSurfaceVolumeList,
                                  const std::vector<const BoundaryVolumeList*>& hBoundaryVolumeList,
                                  VolumeId boundaryVolume,
                                  const GPUQueue& queue) const
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
        unpackedList.fill(MediaList::INVALID_VAL);
        unpackedList[0] = FindVolumeIndex(boundaryVolume);
        unpackedList[1] = FindVolumeIndex(vId);

        std::array<int32_t, MAX_NESTED_MEDIA> priorities;
        priorities[0] = globalVolumeList[unpackedList[0]].second.priority;
        priorities[1] = globalVolumeList[unpackedList[1]].second.priority;
        SortByPrio(unpackedList, priorities, 2);

        MediaList l; l.PackAll(unpackedList);
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
        uint32_t totalCombinations = 1u << bList.size();

        StaticVector<uint32_t, MAX_NESTED_MEDIA> volIndices;
        StaticVector<int32_t, MAX_NESTED_MEDIA> priorities;

        for(VolumeId v : bList)
        {
            volIndices.push_back(FindVolumeIndex(v));
            uint32_t prio = globalVolumeList[volIndices.back()].second.priority;
            priorities.push_back(prio);
        }

        // TODO: We are adding all the singular values
        // here but only singular value should be the boundary
        // one.
        for(uint32_t i = 1; i < totalCombinations; i++)
        {
            std::array<uint32_t, MAX_NESTED_MEDIA> unpackedList;
            std::array<int32_t, MAX_NESTED_MEDIA> curPriorities;
            unpackedList.fill(MediaList::INVALID_VAL);

            uint32_t counter = 0;
            for(uint32_t j = 0; j < MAX_NESTED_MEDIA; j++)
            {
                if(Bit::FetchSubPortion(i, {j, j + 1}) == 0) continue;
                unpackedList[counter] = volIndices[j];
                curPriorities[counter] = priorities[j];
                counter++;
            }
            SortByPrio(unpackedList, curPriorities, counter);

            MediaList l; l.PackAll(unpackedList);
            mediaLists.push_back(l);
        }
    }
    // Send to GPU and let GPU do the hash table stuff.
    DeviceLocalMemory localMem(*queue.Device());
    Span<MediaList> dLists;
    MemAlloc::AllocateMultiData(Tie(dLists), localMem,
                                {mediaLists.size()});
    queue.MemcpyAsync(dLists, Span<const MediaList>(mediaLists));
    queue.IssueWorkKernel<KCPrimeHashTable>
    (
        "KCPrimeHashTable",
        DeviceWorkIssueParams{.workCount = uint32_t(dLists.size())},
        //
        View(),
        dLists
    );
    queue.Barrier().Wait();
}

void MediaTracker::AddNewVolumeToRaysIndirect(// I-O
                                              Span<RayMediaListPack> dPacks,
                                              //
                                              Span<const VolumeIndex> dNewVolumeIndices,
                                              Span<const RayIndex> dRayIndices,
                                              const GPUQueue& queue) const

{
    queue.IssueWorkKernel<KCAddNewVolumeToRayListIndirect>
    (
        "KCAddNewVolumeToRayListIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dRayIndices.size())},
        //
        dPacks,
        View(),
        dNewVolumeIndices,
        dRayIndices
    );
}

void MediaTracker::ResolveCurVolumesOfRaysIndirect(// I-O
                                                   Span<RayMediaListPack> dPacks,
                                                   //
                                                   Span<const RayIndex> dRayIndices,
                                                   const GPUQueue& queue) const
{
    queue.IssueWorkKernel<KCResolveCurVolumesOfRaysIndirect>
    (
        "KCResolveCurVolumesOfRaysIndirect",
        DeviceWorkIssueParams{.workCount = uint32_t(dRayIndices.size())},
        //
        dPacks,
        View(),
        dRayIndices
    );
}