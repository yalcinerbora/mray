#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <map>

#include "Core/Types.h"
#include "Core/MemAlloc.h"

#include "MRayInput/MRayInput.h"

#include "Device/GPUSystem.h"
#include "TracerTypes.h"

// Some common types
// Item and attribute is seperated to support primitives
// Each primitive in a batch may utilize the same attribute so
// "itemCount >= attributeCount"
// For materials for example, it is exactly "itemCount == attributeCount"
template <std::unsigned_integral T>
struct ItemRangeT { Vector<2, T> itemRange; Vector<2, T> attributeRange; };

template <std::unsigned_integral T>
struct ItemCountT { T itemCount; T attributeCount; };

template <std::unsigned_integral T>
using ItemRangeMapT = std::map<T, ItemRangeT<T>>;

template <std::unsigned_integral T>
using ItemCountMapT = std::map<T, ItemCountT<T>>;

template <class GenericGroupType>
concept GenericGroupC = requires(GenericGroupType gg,
                                 MRayInput input,
                                 typename GenericGroupType::IdType id,
                                 const GPUQueue& q)
{
    typename GenericGroupType::IdType;
    typename GenericGroupType::IdInteger;
    typename GenericGroupType::IdList;
    typename GenericGroupType::AttribInfoList;

    {gg.Reserve(std::vector<ItemCountT<typename GenericGroupType::IdInteger>>{})
    } -> std::same_as<typename GenericGroupType::IdList>;
    {gg.CommitReservations()} -> std::same_as<void>;
    {gg.IsInCommitState()} -> std::same_as<bool>;
    {gg.AttributeInfo()
    } -> std::same_as<typename GenericGroupType::AttribInfoList>;
    {gg.PushAttribute(id, uint32_t{}, std::move(input), q)
    } -> std::same_as<void>;
    {gg.PushAttribute(id, Vector2ui{}, uint32_t{}, std::move(input), q)
    } ->std::same_as<void>;
    {gg.PushAttribute(Vector<2, typename GenericGroupType::IdInteger>{},
                      uint32_t{}, std::move(input), q)
    } ->std::same_as<void>;
    {gg.GPUMemoryUsage()} -> std::same_as<size_t>;

    // Can query the type
    {GenericGroupType::TypeName()} -> std::same_as<std::string_view>;
};

template <class IdTypeT, class AttribInfo>
class GenericGroupI
{
    public:
    using IdType            = IdTypeT;
    using IdInteger         = typename IdType::Type;
    using IdList            = std::vector<IdType>;
    using AttribInfoList    = std::vector<AttribInfo>;

    public:
    virtual                 ~GenericGroupI() = default;
    //
    virtual IdList          Reserve(const std::vector<ItemCountT<IdInteger>>&) = 0;
    virtual void            CommitReservations() = 0;
    virtual bool            IsInCommitState() const = 0;
    virtual void            PushAttribute(IdType id,
                                          uint32_t attributeIndex,
                                          MRayInput data,
                                          const GPUQueue& queue) = 0;
    virtual void            PushAttribute(IdType id,
                                          const Vector2ui& subRange,
                                          uint32_t attributeIndex,
                                          MRayInput data,
                                          const GPUQueue& queue) = 0;
    virtual void            PushAttribute(const Vector<2, IdInteger>& idRange,
                                          uint32_t attributeIndex,
                                          MRayInput data,
                                          const GPUQueue& queue) = 0;

    virtual size_t          GPUMemoryUsage() const = 0;
    virtual AttribInfoList  AttributeInfo() const = 0;
};

// Implementation of the common parts
template<class Child, class IdType, class AttribInfo>
class GenericGroupT : public GenericGroupI<IdType, AttribInfo>
{
    static constexpr size_t MapReserveSize = 512;

    public:
    using InterfaceType = GenericGroupI<IdType, AttribInfo>;
    using BaseType      = GenericGroupT<Child, IdType, AttribInfo>;
    using typename InterfaceType::IdList;
    using typename InterfaceType::AttribInfoList;
    using typename InterfaceType::IdInteger;
    using ItemRangeMap  = ItemRangeMapT<IdInteger>;
    using ItemCountMap  = ItemCountMapT<IdInteger>;
    using ItemRange     = ItemRangeT<IdInteger>;
    using ItemCount     = ItemCountT<IdInteger>;

    protected:
    ItemRangeMap        itemRanges;
    ItemCountMap        itemCounts;

    const GPUSystem&    gpuSystem;
    bool                isCommitted;
    IdInteger           groupId;
    DeviceMemory        deviceMem;

    template <class... Args>
    Tuple<Span<Args>...>        GenericCommit(std::array<bool, sizeof...(Args)> isPerItemList);

    template <class T>
    void                        GenericPushData(const Vector<2, IdInteger>& idRange,
                                                const Span<T>& copyRegion,
                                                MRayInput data,
                                                bool isContiguous,
                                                bool isPerItem,
                                                const GPUQueue& queue) const;
    template <class T>
    void                        GenericPushData(IdType id,
                                                const Span<T>& copyRegion,
                                                MRayInput data,
                                                bool isPerItem,
                                                const GPUQueue& queue) const;
    template <class T>
    void                        GenericPushData(IdType id,
                                                const Vector2ui& subRange,
                                                const Span<T>& copyRegion,
                                                MRayInput data,
                                                bool isPerItem,
                                                const GPUQueue& queue) const;

    public:
                                GenericGroupT(uint32_t groupId, const GPUSystem&,
                                              size_t allocationGranularity = 2_MiB,
                                              size_t initialReservartionSize = 32_MiB);
    IdList                      Reserve(const std::vector<ItemCount>&) override;
    virtual bool                IsInCommitState() const override;
    virtual size_t              GPUMemoryUsage() const override;
};

template<class C, class ID, class AI>
template <class... Args>
Tuple<Span<Args>...> GenericGroupT<C, ID, AI>::GenericCommit(std::array<bool, sizeof...(Args)> isPerItemList)
{
    assert(itemRanges.empty());
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s} is in committed state, "
                         " you cannot re-commit!", C::TypeName());
        return Tuple<Span<Args>...>{};
    }
    // Cacluate offsets
    Vector<2, IdInteger> offsets = Vector<2, IdInteger>::Zero();
    for(const auto& c : itemCounts)
    {
        ItemRange range
        {
            .itemRange = Vector<2,IdInteger>(offsets[0], offsets[0] + c.second.itemCount),
            .attributeRange = Vector<2,IdInteger>(offsets[1], offsets[1] + c.second.attributeCount)
        };
        [[maybe_unused]]
        auto r = itemRanges.emplace(c.first, range);
        assert(r.second);

        offsets = Vector<2, IdInteger>(range.itemRange[1], range.attributeRange[1]);
    }
    // Rename for clarity
    Vector<2, IdInteger> totalSize = offsets;

    // Generate offsets etc
    constexpr size_t TotalElements = sizeof...(Args);
    std::array<size_t, TotalElements> sizes;
    for(size_t i = 0; i < TotalElements; i++)
    {
        bool isPerItem = isPerItemList[i];
        sizes[i] = (isPerItem) ? totalSize[0] : totalSize[1];
    }

    using namespace MemAlloc;
    Tuple<Span<Args>...> result;
    result = AllocateMultiData<DeviceMemory, Args...>(deviceMem, sizes);
    isCommitted = true;
    return result;
}


template<class C, class ID, class AI>
template <class T>
void GenericGroupT<C, ID, AI>::GenericPushData(const Vector<2, IdInteger>& idRange,
                                               const Span<T>& copyRegion,
                                               MRayInput data,
                                               bool isContiguous,
                                               bool isPerItem,
                                               const GPUQueue& deviceQueue) const
{
    if(isContiguous)
    {
        size_t count = idRange[1] - idRange[0];
        Span<T> dSubBatch = copyRegion.subspan(idRange[0], count);
        deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
        deviceQueue.IssueBufferForDestruction(std::move(data));
    }
    else
    {
        for(IdInteger i = idRange[0]; i < idRange[1]; i++)
        {
            auto loc = itemRanges.find(i);
            if(loc == itemRanges.end())
            {
                throw MRayError("{:s} id is not found!", C::TypeName());
            }
            Vector<2, IdInteger> r = (isPerItem)
                                        ? loc->second.itemRange
                                        : loc->second.attributeRange;
            size_t count = r[1] - r[0];
            Span<T> dSubBatch = copyRegion.subspan(r[0], count);
            deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
        }
        // Finally issue the buffer for destruction
        deviceQueue.IssueBufferForDestruction(std::move(data));
    }
}

template<class C, class ID, class AI>
template <class T>
void GenericGroupT<C, ID, AI>::GenericPushData(ID id,
                                               const Span<T>& copyRegion,
                                               MRayInput data,
                                               bool isPerItem,
                                               const GPUQueue& deviceQueue) const
{
    const auto it = itemRanges.find(id);
    Vector2ui attribRange = (isPerItem)
                                ? it->second.itemRange
                                : it->second.attributeRange;
    size_t count = attribRange[1] - attribRange[0];
    Span<T> dSubBatch = copyRegion.subspan(attribRange[0], count);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class C, class ID, class AI>
template <class T>
void GenericGroupT<C, ID, AI>::GenericPushData(ID id,
                                               const Vector2ui& subRange,
                                               const Span<T>& copyRegion,
                                               MRayInput data,
                                               bool isPerItem,
                                               const GPUQueue& deviceQueue) const
{
    const auto it = itemRanges.find(id);
    Vector2ui attribRange = (isPerItem)
                                ? it->second.itemRange
                                : it->second.attributeRange;
    size_t count = attribRange[1] - attribRange[0];
    Span<T> dSubBatch = copyRegion.subspan(attribRange[0], count);
    size_t subCount = subRange[1] - subRange[0];
    Span<T> dSubSubBatch = dSubBatch.subspan(subRange[0], subCount);

    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class C, class ID, class AI>
GenericGroupT<C, ID, AI>::GenericGroupT(uint32_t groupId, const GPUSystem& s,
                                        size_t allocationGranularity,
                                        size_t initialReservartionSize)
    : gpuSystem(s)
    , isCommitted(false)
    , groupId(groupId)
    , deviceMem(gpuSystem.AllGPUs(), allocationGranularity, initialReservartionSize)
{
    //itemRanges.reserve(MapReserveSize);
    //itemCounts.reserve(MapReserveSize);
}

template<class C, class ID, class AI>
typename GenericGroupT<C, ID, AI>::IdList GenericGroupT<C, ID, AI>::Reserve(const std::vector<ItemCount>& itemCountList)
{
    if(isCommitted)
    {
        throw MRayError("{:s} is in committed state, "
                        " you change cannot change reservations!",
                        C::TypeName());
    }

    IdList result;
    result.reserve(itemCountList.size());
    for(const ItemCount& i : itemCountList)
    {
        [[maybe_unused]]
        auto r = itemCounts.emplace(static_cast<IdInteger>(itemCounts.size()), i);
        assert(r.second);

        IdInteger innerId  = r.first->first;
        result.push_back(ID::CombinedKey(groupId, innerId));
    }
    return result;
}

template<class C, class ID, class AI>
bool GenericGroupT<C, ID, AI>::IsInCommitState() const
{
    return isCommitted;
}

template<class C, class ID, class AI>
size_t GenericGroupT<C, ID, AI>::GPUMemoryUsage() const
{
    return deviceMem.Size();
}