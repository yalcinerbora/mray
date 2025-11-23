#pragma once

// Media Tracking System for MRay
//
// Main idea for this convoluted approach is that
// Scenes have relatively small media interactions
// compared to the ray count of a wavefront system.
//
// For implementation of
// "Simple Nested Dielectrics in Ray Traced Images",
//
// https://www.researchgate.net/publication/247523037_Simple_Nested_Dielectrics_in_Ray_Traced_Images?enrichId=rgreq-ec5ee58c64ba8c34618adb7501233b7a-XXX&enrichSource=Y292ZXJQYWdlOzI0NzUyMzAzNztBUzoxMDI2MjMxNDc0NjI2NzBAMTQwMTQ3ODY3MzUyNw%3D%3D&el=1_x_3
//
// Each ray need to hold a priority queue / array / stack of the scene's
// maximum nested media count (this system tracks media which is comparable).
//
// I do not have any experience for production scenes but 8 seems generous enough
// to represent almost all of the scenes (This value can be runtime dynamic later).
//
// MRay circulates 2M rays on each iteration, even with 16-bit ids (medium id)
// this all of the arrays total size will be around (2M * 2 * 8) = ~32MiB. Which
// is too much for my taste (and not as scalable since we need to support instanced
// media etc. 16-bit may be too small).
//
// Since we have to pre-allocate the all of the stack for each ray
// (on GPU it is not possible to properly (performantly) alloc such dynamic array),
// maybe not all of the rays hit a that dense of a media chain.
//
// To alleviate this memory, we rely on hash table of chains over enocuntered medium
// chains. This array is statically sized (generously allocated) and ever-filling.
//
// As rays enocunter new media that push/pop their media to the hash table:
//
// Ray:  A->B->C->D / E will be added (Ray Data: HT index of "A->B->C->D", and H)
//
//      - Load "A->B->C->D" via index
//      - Create deterministic order of "A->B->C->D and H" (it is priority order,
//                                                          collisions resolved via mediumId)
//      - Atomically store "A->B->H->C->D" (asuming this is the deterministic order)
//
//
// Ray does not store just the HT index, also two 3-bit integers for the current medium
// index, and resolved media index. This pair will be utilized as interface boundary.
// 3-bit integer corresponds to the chain stored in the HT. If HT index is changed, these
// indices are recalculated. So full bit structure of the ray's "VolumeIndex" is:
//
//   ______________________________
//  | CurMed | NextMed |  HT Index |
//   ------------------------------
// MSB  4-bit    4-bit    24-bit   LSB
//
//
// So we can get away with a single word with these limitations:
//   - Maximum nest is 16
//   - At most 16M medium chains can we stored in the HT.
//
// First one should suffice for all of the scenes that I mentally model. (Execpt for
// maybe some form of recursive snow globe situation (world inside of a world).
//
// Even than we can make this apprach runtime-dynamic with user defined bits etc.
//
// Second one's bound is scary given 16 different nested media. Worst case
// chain combiation is all subsets of these element (-1 for empty set since we always have
// a  media in the list) will be ~2^16. But it is probably not possible to reach
// all of the subsets of these 16 media with plausible scenes (I could not mentally model this
// so it may not be true).
//
// So we allocate around 2 MiB of HT with 8 chains (although index structre supports it
// 8 is used to reduce combinations to 256.
//
// So systems assumes as the worst case of:
//  - 256 interactions per set of interactions (for example, glass/ ice cube / water
//                                              combination is a set)
//  - 256 different sets (glass with orange juice ice cube, another glass with wine,
//                        clouds etc. are all different sets)
//  - Each set at most nest to 8 different media.
//
// This results in to 2MiB of memory. So we save around 16x memory (32x with uint32_t indies)
// Also, if interaction is shallow and since this is a HT, set of interactions
// automatically increase. If you have a cloud, you can have 1024 of these clouds
// instanced. This is somehwat of a limitation though.

#include "Core/Types.h"
#include "Core/TracerAttribInfo.h"

#include "Device/GPUAtomic.h"

#include "Random.h"
#include "TracerTypes.h"

// Reducing memory here to set the indices
//
inline constexpr uint32_t MAX_MEDIA_BITS   = 20;
inline constexpr uint32_t MAX_NESTED_MEDIA = 8;
inline constexpr uint32_t MAX_MEDIA_COUNT  = 1 << MAX_MEDIA_BITS;
inline constexpr uint32_t MEDIA_LIST_BITS = MAX_MEDIA_BITS * MAX_NESTED_MEDIA;
inline constexpr uint32_t MEDIA_LIST_WORDS = MEDIA_LIST_BITS / (sizeof(uint32_t) * CHAR_BIT);
static_assert(MEDIA_LIST_BITS % (sizeof(uint32_t) * CHAR_BIT) == 0,
              "Wasted bits in MediaTracker!");

struct MediaList
{
    using IndexList = std::array<uint32_t, MAX_NESTED_MEDIA>;
    using PackedList = std::array<uint32_t, MEDIA_LIST_WORDS>;

    PackedList listRaw;
    // Functionality to lift the data to register space
    // TODO: This may be a waste, we did not implemented the
    // usage yet
    MR_PF_DECL   IndexList Unpack() const;
    MR_PF_DECL_V void      Pack(const IndexList&);

    auto operator<=>(const MediaList&) const = default;
};

class RayMediaListPack
{
    // Utilizing the TriKey here to save impl.
    // Internal Index
    static constexpr uint32_t ENTER_EXIT_BIT = 1;
    static constexpr uint32_t PASSTHROUGH_BIT = 1;
    static constexpr uint32_t INNER_INDEX_BITS = Bit::RequiredBitsToRepresent(MAX_NESTED_MEDIA - 1);
    static constexpr uint32_t OUTER_INDEX_BITS = ((sizeof(uint32_t) * CHAR_BIT) -
                                                  INNER_INDEX_BITS * 2 -
                                                  ENTER_EXIT_BIT -
                                                  PASSTHROUGH_BIT);
    //
    static constexpr Vector2ui OUTER_INDEX_RANGE = Vector2ui(0, OUTER_INDEX_BITS);
    static constexpr Vector2ui CUR_INDEX_RANGE   = Vector2ui(OUTER_INDEX_RANGE[1],
                                                             OUTER_INDEX_RANGE[1] + INNER_INDEX_BITS);
    static constexpr Vector2ui NEXT_INDEX_RANGE  = Vector2ui(CUR_INDEX_RANGE[1],
                                                             CUR_INDEX_RANGE[1] + INNER_INDEX_BITS);
    static constexpr Vector2ui IS_ENTERING_RANGE = Vector2ui(NEXT_INDEX_RANGE[1],
                                                             NEXT_INDEX_RANGE[1] + ENTER_EXIT_BIT);
    static constexpr Vector2ui PASSTHROUGH_RANGE = Vector2ui(IS_ENTERING_RANGE[1],
                                                             IS_ENTERING_RANGE[1] + PASSTHROUGH_BIT);
    static_assert(PASSTHROUGH_RANGE[1] == (sizeof(uint32_t) * CHAR_BIT));


    private:
    uint32_t pack;

    public:
    MR_PF_DECL_V        RayMediaListPack(uint32_t curMed, uint32_t nextMed, uint32_t outer);

    MR_PF_DECL uint32_t CurMediaIndex() const;
    MR_PF_DECL uint32_t NextMediaIndex() const;
    MR_PF_DECL uint32_t OuterIndex() const;
    MR_PF_DECL bool     IsEntering() const;
    MR_PF_DECL bool     IsRayPassedThrough() const;
    //
    MR_PF_DECL_V void   SetCurMediaIndex(uint32_t);
    MR_PF_DECL_V void   SetNextMediaIndex(uint32_t);
    MR_PF_DECL_V void   SetOuterIndex(uint32_t);
    MR_PF_DECL_V void   SetEntering(bool);
    MR_PF_DECL_V void   SetRayPassedThrough(bool);
};

class MediaTrackerView
{
    // TODO: This could be 16-bit but my 1080 does not support
    // it. Need to check 2000 series (which should be the bare minimum).
    using LockInt = uint32_t;

    public:
    static constexpr auto EMPTY_VAL    = uint32_t(UINT32_MAX);
    static constexpr auto LOCK_VAL     = LockInt(1);
    static constexpr auto UNLOCKED_VAL = LockInt(0);
    struct InsertResult
    {
        uint32_t index;
        bool     isInserted;
    };

    private:
    Span<const VolumeKeyPack> dGlobalVolumeList;
    Span<MediaList>           dValues;
    Span<LockInt>             dLocks;



    public:
    // Constructors & Destructor
              MediaTrackerView() = default;
    MRAY_HOST MediaTrackerView(Span<const VolumeKeyPack> dGlobalVolumeList,
                               Span<MediaList> dValues,
                               Span<LockInt> dLocks);

    //
    MR_GF_DECL
    InsertResult TryInsertAtomic(const MediaList& list) const;

    MR_GF_DECL
    void UpdateRayMediaList(RayMediaListPack& rayMediaListIndex,
                            VolumeIndex nextVolumeIndex) const;

    MR_HF_DECL
    VolumeKeyPack GetVolumeKeyPack(RayMediaListPack rayMediaListIndex) const;
};

class MediaTracker
{
    using VolumeList = std::vector<Pair<VolumeId, VolumeKeyPack>>;

    private:
    const GPUSystem& gpuSystem;
    const VolumeList& globalVolumeList;
    // Memory
    DeviceMemory        mem;
    Span<VolumeKeyPack> dGlobalVolumeList;
    Span<uint32_t>      dLocksHT;
    Span<MediaList>     dMediaListHT;

    uint32_t FindVolumeIndex(VolumeId) const;

    public:
    // Constructors & Destructor
    MediaTracker(const VolumeList& globalVolumeList,
                 uint32_t maximumEntryCount,
                 const GPUSystem& gpuSystem);


    void SetStartingVolumeIndirect(// Output
                                   Span<RayMediaListPack> packs,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   // Constants
                                   const BoundaryVolumeList& nestedVolumeList,
                                   const GPUQueue&);

    void PrimeHashTable(const std::vector<const SurfaceVolumeList*>& hSurfaceVolumeList,
                        const std::vector<const BoundaryVolumeList*>& hBoundaryVolumeList,
                        VolumeId boundaryVolume,
                        const GPUQueue&);

    // Misc
    size_t           GPUMemoryUsage() const;
    MediaTrackerView View() const;

    //void ResolveInterface(// I-O
    //                      Span<RayMediaListPack> packs,
    //                      const GPUQueue&);

    //void HashForPartition(Span<const RayMediaListPack> packs);
};

MR_PF_DEF
uint32_t RayMediaListPack::CurMediaIndex() const
{
    return Bit::FetchSubPortion(pack, {CUR_INDEX_RANGE[0], CUR_INDEX_RANGE[1]});
}

MR_PF_DEF
uint32_t RayMediaListPack::NextMediaIndex() const
{
    return Bit::FetchSubPortion(pack, {NEXT_INDEX_RANGE[0], NEXT_INDEX_RANGE[1]});
}

MR_PF_DEF
uint32_t RayMediaListPack::OuterIndex() const
{
    return Bit::FetchSubPortion(pack, {OUTER_INDEX_RANGE[0], OUTER_INDEX_RANGE[1]});
}
MR_PF_DEF
bool RayMediaListPack::IsEntering() const
{
    return bool(Bit::FetchSubPortion(pack, {IS_ENTERING_RANGE[0], IS_ENTERING_RANGE[1]}));
}

MR_PF_DEF
bool RayMediaListPack::IsRayPassedThrough() const
{
    return bool(Bit::FetchSubPortion(pack, {PASSTHROUGH_RANGE[0], PASSTHROUGH_RANGE[1]}));
}

MR_PF_DEF_V
void RayMediaListPack::SetCurMediaIndex(uint32_t i)
{
    pack = Bit::SetSubPortion(pack, i, {CUR_INDEX_RANGE[0], CUR_INDEX_RANGE[1]});
}

MR_PF_DEF_V
void RayMediaListPack::SetNextMediaIndex(uint32_t i)
{
    pack = Bit::SetSubPortion(pack, i, {NEXT_INDEX_RANGE[0], NEXT_INDEX_RANGE[1]});
}

MR_PF_DEF_V
void RayMediaListPack::SetOuterIndex(uint32_t i)
{
    pack = Bit::SetSubPortion(pack, i, {OUTER_INDEX_RANGE[0], OUTER_INDEX_RANGE[1]});
}

MR_PF_DEF_V
void RayMediaListPack::SetEntering(bool b)
{
    pack = Bit::SetSubPortion(pack, b ? 1 : 0,
                              {IS_ENTERING_RANGE[0], IS_ENTERING_RANGE[1]});
}

MR_PF_DEF_V
void RayMediaListPack::SetRayPassedThrough(bool b)
{
    pack = Bit::SetSubPortion(pack, b ? 1 : 0,
                              {PASSTHROUGH_RANGE[0], PASSTHROUGH_RANGE[1]});
}

MR_PF_DEF
typename MediaList::IndexList
MediaList::Unpack() const
{
    // TODO: I just hand rolled the fetch routines
    // Did not bother generating a dynamic code.
    static_assert(MAX_MEDIA_BITS == 20 && MAX_NESTED_MEDIA == 8,
                  "The code statically rolled for these values only!");
    //
    IndexList result;
    // 0
    result[0] = Bit::FetchSubPortion(listRaw[0], {0, 20});
    // 1
    result[1] = Bit::Compose<12, 8>
    (
        Bit::FetchSubPortion(listRaw[0], {20, 32}),
        Bit::FetchSubPortion(listRaw[1], {0 , 8})
    );
    // 2
    result[2] = Bit::FetchSubPortion(listRaw[1], {8, 28});
    // 3
    result[3] = Bit::Compose<4, 16>
    (
        Bit::FetchSubPortion(listRaw[1], {28, 32}),
        Bit::FetchSubPortion(listRaw[2], {0 , 16})
    );
    // 4
    result[4] = Bit::Compose<16, 4>
    (
        Bit::FetchSubPortion(listRaw[2], {16, 32}),
        Bit::FetchSubPortion(listRaw[3], {0 ,  4})
    );
    // 5
    result[5] = Bit::FetchSubPortion(listRaw[3], {4, 24});
    // 6
    result[6] = Bit::Compose<8, 12>
    (
        Bit::FetchSubPortion(listRaw[3], {24, 32}),
        Bit::FetchSubPortion(listRaw[4], {0 ,  12})
    );
    // 7
    result[7] = Bit::FetchSubPortion(listRaw[4], {12, 32});
    return result;
}

MR_PF_DEF_V
void MediaList::Pack(const IndexList& list)
{
    static_assert(MAX_MEDIA_BITS == 20 && MAX_NESTED_MEDIA == 8,
                  "The code statically rolled for these values only!");
    // 0
    listRaw[0] = Bit::SetSubPortion(listRaw[0], list[0]      , { 0, 20});
    // 1
    listRaw[0] = Bit::SetSubPortion(listRaw[0], list[1]      , {20, 32});
    listRaw[1] = Bit::SetSubPortion(listRaw[1], list[1] >> 12, { 0, 8});
    // 2
    listRaw[1] = Bit::SetSubPortion(listRaw[1], list[2]      , { 8, 28});
    // 3
    listRaw[1] = Bit::SetSubPortion(listRaw[1], list[3]      , {28, 32});
    listRaw[2] = Bit::SetSubPortion(listRaw[2], list[3] >>  4, { 0, 16});
    // 4
    listRaw[2] = Bit::SetSubPortion(listRaw[2], list[4]      , {16, 32});
    listRaw[3] = Bit::SetSubPortion(listRaw[3], list[4] >> 16, { 0, 4});
    // 5
    listRaw[3] = Bit::SetSubPortion(listRaw[3], list[5]      , { 4, 24});
    // 6
    listRaw[3] = Bit::SetSubPortion(listRaw[3], list[6]      , {24, 32});
    listRaw[4] = Bit::SetSubPortion(listRaw[4], list[6] >>  8, { 0, 12});
    // 7
    listRaw[7] = Bit::SetSubPortion(listRaw[4], list[7]      , {12, 32});
}

MRAY_HOST
inline
MediaTrackerView::MediaTrackerView(Span<const VolumeKeyPack> dGlobalVolumeList,
                                   Span<MediaList> dValues,
                                   Span<LockInt> dLocks)
    : dGlobalVolumeList(dGlobalVolumeList)
    , dValues(dValues)
    , dLocks(dLocks)
{}

MR_GF_DEF
typename MediaTrackerView::InsertResult
MediaTrackerView::TryInsertAtomic(const MediaList& list) const
{
    // CUDA stuff, it does not like gobal space constexpr
    // assumes it is on host
    static constexpr auto EMPTY    = EMPTY_VAL;
    static constexpr auto LOCKED   = LOCK_VAL;
    static constexpr auto UNLOCKED = UNLOCKED_VAL;

    using RNGFunctions::HashPCG64::Hash;
    static_assert(MEDIA_LIST_WORDS == 5u, "This part of the code is static! Add/Remove hashes");
    uint32_t h = uint32_t(Hash(list.listRaw[0], list.listRaw[1],
                               list.listRaw[2], list.listRaw[3],
                               list.listRaw[4]));
    if(h == EMPTY_VAL) h++;

    uint32_t tableSize = dValues.size();
    uint32_t divMask = uint32_t(tableSize - 1);
    assert(Bit::PopC(tableSize) == 1);

    // Hash table with linear probing
    uint32_t index = h & divMask;
    for(uint32_t i = index;; i = (i + 1) & divMask)
    {
        // We will try to do atomic CAS lock shortly but we can
        // pre-check the lock here, if it is unlocked
        // writer already done writing all the indices.
        if(dLocks[i] == UNLOCKED && dValues[i] == list)
            return InsertResult{i, false};

        // Now the tricky part
        // Obviously device does not have 5x32-bit atomic operation,
        // (Although new NVIDIA devices has 128-bit atomics)
        // So we store hash list as both as a locking mechanism
        // and to fast check if we continue probing or not.
        // ============ //
        //   ACQ. LOCK  //
        // ============ //
        using namespace DeviceAtomic;
        uint32_t old;
        do
        {
            old = AtomicCompSwap(dLocks[i], UNLOCKED, LOCKED);
        }
        while(old == LOCK_VAL);
        ThreadFenceGrid();

        // Now we are fine (hopefully)
        if(dValues[i].listRaw[0] == EMPTY)
        {
            // Successfull Insert
            dValues[i] = list;
            // ============ //
            //  REL. LOCK   //
            // ============ //
            ThreadFenceGrid();
            AtomicStore(dLocks[i], UNLOCKED);
            return InsertResult{i, true};
        }
        else if(dValues[i] == list)
        {
            // Unsuccessfull insert but we find the data
            // ============ //
            //  REL. LOCK   //
            // ============ //
            AtomicStore(dLocks[i], UNLOCKED);
            return InsertResult{i, false};
        }
        // Occupied
        // ============ //
        //  REL. LOCK   //
        // ============ //
        AtomicStore(dLocks[i], UNLOCKED);
        //
        if(i == index - 1) break;
    }
    return InsertResult{0, false};
}

MR_GF_DEF
void MediaTrackerView::UpdateRayMediaList(RayMediaListPack& rayMediaListIndex,
                                          VolumeIndex nextVolumeIndexPack) const
{
    uint32_t htIndex = rayMediaListIndex.OuterIndex();
    MediaList curList = dValues[htIndex];

    // Add new volume to the stack
    using UnpackedList = typename MediaList::IndexList;
    UnpackedList volIndices = curList.Unpack();

    // Acquire the index
    uint32_t newNextVolInnerIndex;


    // Sorted Add
    uint32_t nextVolumeIndex = nextVolumeIndexPack.FetchIndexPortion();
    uint32_t liftedVolIndex = nextVolumeIndex;
    uint32_t liftPrio = dGlobalVolumeList[nextVolumeIndex].priority;
    CommonKey liftKey = CommonKey(dGlobalVolumeList[nextVolumeIndex].medKey);
    for(uint32_t i = 0; i < MAX_NESTED_MEDIA; i++)
    {
        uint32_t curPrio = dGlobalVolumeList[volIndices[i]].priority;
        CommonKey curKey = CommonKey(dGlobalVolumeList[volIndices[i]].medKey);

        if(volIndices[i] == EMPTY_VAL)
        {
            volIndices[i] = liftedVolIndex;
            i++;
            break;
        }
        else if((liftPrio > curPrio) || (liftPrio == curPrio && liftKey > curKey))
        {
            if(liftedVolIndex == nextVolumeIndex)
                newNextVolInnerIndex = i;

            std::swap(liftedVolIndex, volIndices[i]);
            std::swap(liftPrio, curPrio);
            std::swap(liftKey, curKey);
        }
    }

    uint32_t newCurVolInnerIndex = rayMediaListIndex.CurMediaIndex();
    if(newNextVolInnerIndex <= newCurVolInnerIndex)
        newCurVolInnerIndex++;


    // Finally Pack and Insert
    curList.Pack(volIndices);
    auto [newIndex, _] = TryInsertAtomic(curList);

    // Find the next volume and cur volume index
    rayMediaListIndex.SetOuterIndex(newIndex);
    rayMediaListIndex.SetCurMediaIndex(newCurVolInnerIndex);
    rayMediaListIndex.SetNextMediaIndex(newNextVolInnerIndex);
    rayMediaListIndex.SetEntering(nextVolumeIndexPack.FetchBatchPortion());
}

MR_HF_DEF
VolumeKeyPack
MediaTrackerView::GetVolumeKeyPack(RayMediaListPack) const
{
    return VolumeKeyPack{};
}

inline
MediaTrackerView MediaTracker::View() const
{
    return MediaTrackerView(dGlobalVolumeList,
                            dMediaListHT,
                            dLocksHT);
}

inline
size_t MediaTracker::GPUMemoryUsage() const
{
    return mem.Size();
}