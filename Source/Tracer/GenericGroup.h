#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>

#include "Core/Types.h"
#include "Core/Map.h"
#include "Core/MemAlloc.h"
#include "Core/DataStructures.h"
#include "Core/TracerI.h"
#include "Core/Error.h"

#include "TransientPool/TransientPool.h"

#include "Device/GPUSystemForward.h"

#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "TextureView.h"

using AttributeRanges = StaticVector<Vector<2, size_t>,
                                     TracerConstants::MaxAttributePerGroup>;

template <class GenericGroupType>
concept GenericGroupC = requires(GenericGroupType gg,
                                 TransientData input,
                                 typename GenericGroupType::Id id,
                                 const GPUQueue& q)
{
    typename GenericGroupType::Id;
    typename GenericGroupType::IdInt;
    typename GenericGroupType::IdList;
    typename GenericGroupType::AttribInfoList;

    {gg.Reserve(std::vector<AttributeCountList>{})
    } -> std::same_as<typename GenericGroupType::IdList>;
    {gg.CommitReservations()} -> std::same_as<void>;
    {gg.IsInCommitState()} -> std::same_as<bool>;
    {gg.AttributeInfo()
    } -> std::same_as<typename GenericGroupType::AttribInfoList>;
    {gg.PushAttribute(id, uint32_t{}, std::move(input), q)
    } -> std::same_as<void>;
    {gg.PushAttribute(id, uint32_t{}, Vector2ui{}, std::move(input), q)
    } ->std::same_as<void>;
    {gg.PushAttribute(id, id, uint32_t{}, std::move(input), q)
    } ->std::same_as<void>;
    {gg.GPUMemoryUsage()} -> std::same_as<size_t>;

    // Can query the type
    {gg.Name()} -> std::same_as<std::string_view>;
    {GenericGroupType::TypeName()} -> std::same_as<std::string_view>;
};

template <class IdTypeT, class AttribInfoT>
class GenericGroupI
{
    public:
    using Id                = IdTypeT;
    using IdInt             = typename Id::Type;
    using IdList            = std::vector<Id>;
    using AttribInfo        = AttribInfoT;
    using AttribInfoList    = StaticVector<AttribInfo, TracerConstants::MaxAttributePerGroup>;

    public:
    virtual                 ~GenericGroupI() = default;
    //
    virtual IdList          Reserve(const std::vector<AttributeCountList>&) = 0;
    virtual void            CommitReservations() = 0;
    virtual bool            IsInCommitState() const = 0;
    virtual void            PushAttribute(Id id, uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) = 0;
    virtual void            PushAttribute(Id id, uint32_t attributeIndex,
                                          const Vector2ui& subRange,
                                          TransientData data,
                                          const GPUQueue& queue) = 0;
    virtual void            PushAttribute(Id idStart, Id idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) = 0;
    // Finalize functionality,
    // This will be called just before commit surfaces, once.
    virtual void                Finalize(const GPUQueue&) = 0;
    //
    virtual size_t              GPUMemoryUsage() const = 0;
    virtual AttribInfoList      AttributeInfo() const = 0;
    virtual std::string_view    Name() const = 0;
    virtual IdInt               GroupId() const = 0;
};

// Implementation of the common parts
template<class IdTypeT, class AttribInfoT>
class GenericGroupT : public GenericGroupI<IdTypeT, AttribInfoT>
{
    static constexpr size_t MapReserveSize = 512;

    public:
    using InterfaceType = GenericGroupI<IdTypeT, AttribInfoT>;
    using BaseType      = GenericGroupT<IdTypeT, AttribInfoT>;
    using typename InterfaceType::Id;
    using typename InterfaceType::IdInt;
    using typename InterfaceType::IdList;
    using typename InterfaceType::AttribInfo;
    using typename InterfaceType::AttribInfoList;

    using ItemRangeMap  = Map<IdInt, AttributeRanges>;
    using ItemCountMap  = Map<IdInt, AttributeCountList>;

    protected:
    ItemRangeMap        itemRanges;
    ItemCountMap        itemCounts;

    const GPUSystem&    gpuSystem;
    std::atomic_bool    isCommitted;
    std::mutex          mutex;
    IdInt               groupId;
    DeviceMemory        deviceMem;


    const AttributeRanges&      FindRange(IdInt) const;

    template <class... Args>
    Tuple<Span<Args>...>    GenericCommit(std::array<size_t, sizeof...(Args)> countLookup);

    template <class T>
    void GenericPushData(const Span<T>& dAttributeRegion,
                         //
                         IdInt id, uint32_t attribIndex,
                         TransientData data,
                         const GPUQueue& deviceQueue) const;
    template <class T>
    void GenericPushData(const Span<T>& dAttributeRegion,
                         //
                         Vector<2, IdInt> idRange,
                         uint32_t attribIndex,
                         TransientData data,
                         const GPUQueue& deviceQueue) const;
    template <class T>
    void GenericPushData(const Span<T>& dAttributeRegion,
                         //
                         IdInt id, uint32_t attribIndex,
                         const Vector2ui& subRange,
                         TransientData data,
                         const GPUQueue& deviceQueue) const;

    public:
            GenericGroupT(uint32_t groupId, const GPUSystem&,
                          size_t allocationGranularity = 2_MiB,
                          size_t initialReservartionSize = 4_MiB);
    IdList  Reserve(const std::vector<AttributeCountList>&) override;
    bool    IsInCommitState() const override;
    size_t  GPUMemoryUsage() const override;
    IdInt   GroupId() const override;
    // Finalize is usefull on rare occasions so we default it to empty
    void    Finalize(const GPUQueue&) override {};
};

template<class IdType, class AttributeInfoType>
class GenericTexturedGroupT : public GenericGroupT<IdType, AttributeInfoType>
{
    using Parent = GenericGroupT<IdType, AttributeInfoType>;

    template<uint32_t D, class T>
    std::vector<TracerTexView<D, T>>
            ConvertToView(std::vector<TextureId> texIds,
                          uint32_t attributeIndex) const;

    template<uint32_t D, class T>
    std::vector<Optional<TracerTexView<D, T>>>
            ConvertToView(std::vector<Optional<TextureId>> texIds,
                          uint32_t attributeIndex) const;

    protected:
    const TextureViewMap& globalTextureViews;

    template<uint32_t D, class T>
    void            GenericPushTexAttribute(Span<ParamVaryingData<D, T>>,
                                            //
                                            IdType idStart, IdType idEnd,
                                            uint32_t attributeIndex,
                                            TransientData,
                                            std::vector<Optional<TextureId>>,
                                            const GPUQueue& queue);
    template<uint32_t D, class T>
    void            GenericPushTexAttribute(Span<Optional<TracerTexView<D, T>>>,
                                            //
                                            IdType idStart, IdType idEnd,
                                            uint32_t attributeIndex,
                                            std::vector<Optional<TextureId>>,
                                            const GPUQueue& queue);
    template<uint32_t D, class T>
    void            GenericPushTexAttribute(Span<TracerTexView<D, T>>,
                                            //
                                            IdType idStart, IdType idEnd,
                                            uint32_t attributeIndex,
                                            std::vector<TextureId>,
                                            const GPUQueue& queue);

    public:
    // Constructors & Destructor
    GenericTexturedGroupT(uint32_t groupId, const GPUSystem&,
                          const TextureViewMap&,
                          size_t allocationGranularity = 2_MiB,
                          size_t initialReservartionSize = 4_MiB);

    // Extra textured functionality
    virtual void    PushTexAttribute(IdType idStart, IdType idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) = 0;
    virtual void    PushTexAttribute(IdType idStart, IdType idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) = 0;
    virtual void    PushTexAttribute(IdType idStart, IdType idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) = 0;
};

template<class ID, class AI>
const AttributeRanges& GenericGroupT<ID, AI>::FindRange(IdInt id) const
{
    auto range = itemRanges.at(ID(id).FetchIndexPortion());
    if(!range)
    {
        throw MRayError("{:s}:{:d}: Unkown key {}",
                        this->Name(), this->groupId, id);
    }
    return range.value().get();
}

template<class ID, class AI>
template <class... Args>
Tuple<Span<Args>...> GenericGroupT<ID, AI>::GenericCommit(std::array<size_t, sizeof...(Args)> countLookup)
{
    constexpr size_t TypeCount = sizeof...(Args);
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s}:{:d}: is in committed state, "
                         "you cannot re-commit!", this->Name(), groupId);
        return Tuple<Span<Args>...>{};
    }
    // Cacluate offsets
    std::array<size_t, TypeCount> offsets = {0ull};
    for(const auto& c : itemCounts)
    {
        const auto& counts = c.second;
        AttributeRanges range;
        for(size_t i = 0; i < TypeCount; i++)
        {
            size_t count = counts[countLookup[i]];
            range.emplace_back(offsets[i], offsets[i] + count);
            offsets[i] += count;
        }

        [[maybe_unused]]
        auto r = itemRanges.emplace(c.first, range);
        assert(r.second);
    }
    // Rename for clarity
    const auto& totalSize = offsets;
    if(std::reduce(totalSize.cbegin(), totalSize.cend()) == 0)
    {
        MRAY_WARNING_LOG("{:s}:{:d}: committing as empty, "
                         "is this correct?", this->Name(), groupId);
        isCommitted = true;
        return Tuple<Span<Args>...>{};
    }

    using namespace MemAlloc;
    Tuple<Span<Args>...> result;
    result = AllocateMultiData<DeviceMemory, Args...>(deviceMem, totalSize);
    isCommitted = true;
    return result;
}

template<class ID, class AI>
template <class T>
void GenericGroupT<ID, AI>::GenericPushData(const Span<T>& dAttributeRegion,
                                            //
                                            IdInt id, uint32_t attribIndex,
                                            TransientData data,
                                            const GPUQueue& deviceQueue) const
{
    assert(data.IsFull());
    auto range = FindRange(id)[attribIndex];
    size_t itemCount = range[1] - range[0];
    assert(data.Size<T>() == itemCount);

    Span<T> dSubBatch = dAttributeRegion.subspan(range[0], itemCount);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class ID, class AI>
template <class T>
void GenericGroupT<ID, AI>::GenericPushData(const Span<T>& dAttributeRegion,
                                            //
                                            Vector<2, IdInt> idRange,
                                            uint32_t attribIndex,
                                            TransientData data,
                                            const GPUQueue& deviceQueue) const
{
    assert(data.IsFull());
    auto rangeStart = FindRange(idRange[0])[attribIndex];
    auto rangeEnd   = FindRange(idRange[1])[attribIndex];
    size_t itemCount = rangeEnd[1] - rangeStart[0];
    assert(data.Size<T>() == itemCount);

    Span<T> dSubBatch = dAttributeRegion.subspan(rangeStart[0], itemCount);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class ID, class AI>
template <class T>
void GenericGroupT<ID, AI>::GenericPushData(const Span<T>& dAttributeRegion,
                                            //
                                            IdInt id, uint32_t attribIndex,
                                            const Vector2ui& subRange,
                                            TransientData data,
                                            const GPUQueue& deviceQueue) const
{
    assert(data.IsFull());
    auto range = FindRange(id)[attribIndex];
    size_t itemCount = subRange[1] - subRange[0];
    assert(data.Size<T>() <= itemCount);

    auto dLocalSpan = dAttributeRegion.subspan(range[0], range[1] - range[0]);
    auto dLocalSubspan = dLocalSpan.subspan(subRange[0], itemCount);
    deviceQueue.MemcpyAsync(dLocalSubspan, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class ID, class AI>
GenericGroupT<ID, AI>::GenericGroupT(uint32_t groupId, const GPUSystem& s,
                                     size_t allocationGranularity,
                                     size_t initialReservartionSize)
    : gpuSystem(s)
    , isCommitted(false)
    , groupId(groupId)
    , deviceMem(gpuSystem.AllGPUs(), allocationGranularity, initialReservartionSize)
{}

template<class ID, class AI>
typename GenericGroupT<ID, AI>::IdList
GenericGroupT<ID, AI>::Reserve(const std::vector<AttributeCountList>& countArrayList)
{
    std::lock_guard<std::mutex> lock(mutex);
    assert(!countArrayList.empty());
    if(isCommitted)
    {
        throw MRayError("{:s}:{:d}: is in committed state, "
                        "you cannot change reservations!",
                        this->Name(), groupId);
    }
    // Lets not use zero
    IdInt lastId = (itemCounts.empty()) ? 0 : std::prev(itemCounts.end())->first + 1;
    IdList result(countArrayList.size());
    for(size_t i = 0; i < countArrayList.size(); i++)
    {
        IdInt id = lastId++;

        [[maybe_unused]]
        auto r = itemCounts.emplace(id, countArrayList[i]);
        assert(r.second);

        // Convert result to actual groupId packed id
        result[i] = ID::CombinedKey(groupId, id);
    }
    return result;
}

template<class ID, class AI>
bool GenericGroupT<ID, AI>::IsInCommitState() const
{
    return isCommitted;
}

template<class ID, class AI>
size_t GenericGroupT<ID, AI>::GPUMemoryUsage() const
{
    return deviceMem.Size();
}

template<class ID, class AI>
typename GenericGroupT<ID, AI>::IdInt
GenericGroupT<ID, AI>::GroupId() const
{
    return groupId;
}

template<class I, class A>
template<uint32_t D, class T>
std::vector<TracerTexView<D, T>>
GenericTexturedGroupT<I, A>::ConvertToView(std::vector<TextureId> texIds,
                                           uint32_t attributeIndex) const
{
    using ViewType = TracerTexView<D, T>;
    std::vector<ViewType> result;
    result.reserve(texIds.size());
    for(const auto& texId : texIds)
    {
        auto optView = globalTextureViews.at(texId);
        if(!optView)
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) is not found",
                            this->Name(), this->groupId,
                            static_cast<uint32_t>(texId));
        }
        const GenericTextureView& view = optView.value();
        if(!std::holds_alternative<ViewType>(view))
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) does not have "
                            "a correct type for, Attribute {:d}",
                            this->Name(), this->groupId,
                            static_cast<uint32_t>(texId), attributeIndex);
            return std::vector<Optional<TracerTexView<D, T>>>{};
        }
        result.push_back(std::get<ViewType>(view));
    }
    return result;
}

template<class I, class A>
template<uint32_t D, class T>
std::vector<Optional<TracerTexView<D, T>>>
GenericTexturedGroupT<I, A>::ConvertToView(std::vector<Optional<TextureId>> texIds,
                                           uint32_t attributeIndex) const
{
    using ViewType = TracerTexView<D, T>;

    std::vector<Optional<ViewType>> result;
    result.reserve(texIds.size());
    for(const auto& texId : texIds)
    {
        if(!texId)
        {
            result.push_back(std::nullopt);
            continue;
        }
        auto optView = globalTextureViews.at(texId.value());
        if(!optView)
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) is not found",
                            this->Name(), this->groupId,
                            static_cast<uint32_t>(texId.value()));
        }
        const GenericTextureView& view = optView.value();
        if(!std::holds_alternative<ViewType>(view))
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) does not have "
                            "a correct type for, Attribute {:d}",
                            this->Name(), this->groupId,
                            static_cast<uint32_t>(texId.value()),
                            attributeIndex);
        }
        result.push_back(std::get<ViewType>(view));
    }
    return result;
}


template<class I, class A>
GenericTexturedGroupT<I, A>::GenericTexturedGroupT(uint32_t groupId, const GPUSystem& s,
                                                   const TextureViewMap& map,
                                                   size_t allocationGranularity,
                                                   size_t initialReservartionSize)
    : Parent(groupId, s,
             allocationGranularity,
             initialReservartionSize)
    , globalTextureViews(map)
{}

template<class I, class A>
template<uint32_t D, class T>
void GenericTexturedGroupT<I, A>::GenericPushTexAttribute(Span<ParamVaryingData<D, T>> dAttributeSpan,
                                                          //
                                                          I idStart, I idEnd,
                                                          uint32_t attributeIndex,
                                                          TransientData hData,
                                                          std::vector<Optional<TextureId>> optionalTexIds,
                                                          const GPUQueue& queue)
{
    assert(hData.IsFull());
    auto hOptTexViews = ConvertToView<D, T>(std::move(optionalTexIds),
                                            attributeIndex);

    // Now we need to be careful
    auto rangeStart = this->FindRange(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->FindRange(idEnd.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<ParamVaryingData<D, T>> dSubspan = dAttributeSpan.subspan(rangeStart[0], count);

    assert(dSubspan.size() == hOptTexViews.size());
    assert(hOptTexViews.size() == hData.Size<T>());

    // Construct in host, then memcpy
    std::vector<ParamVaryingData<D, T>> hParamVaryingData;
    hParamVaryingData.reserve(dSubspan.size());
    Span<const T> hDataSpan = hData.AccessAs<T>();
    for(size_t i = 0; i < hOptTexViews.size(); i++)
    {
        auto pvd = hOptTexViews[i].has_value()
            ? ParamVaryingData<D, T>(hOptTexViews[i].value())
            : ParamVaryingData<D, T>(hDataSpan[i]);
        hParamVaryingData.push_back(pvd);
    }
    auto hParamVaryingDataSpan = Span<const ParamVaryingData<D, T>>(hParamVaryingData.cbegin(),
                                                                    hParamVaryingData.cend());
    queue.MemcpyAsync(dSubspan, hParamVaryingDataSpan);
    // TODO: Try to find a way to remove this wait
    queue.Barrier().Wait();

}

template<class I, class A>
template<uint32_t D, class T>
void GenericTexturedGroupT<I, A>::GenericPushTexAttribute(Span<Optional<TracerTexView<D, T>>> dAttributeSpan,
                                                          //
                                                          I idStart, I idEnd,
                                                          uint32_t attributeIndex,
                                                          std::vector<Optional<TextureId>> optionalTexIds,
                                                          const GPUQueue& queue)
{
    auto hOptTexViews = ConvertToView<D, T>(std::move(optionalTexIds),
                                            attributeIndex);

    // YOLO memcpy here! Hopefully it works
    auto rangeStart = this->FindRange(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->FindRange(idEnd.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<Optional<TracerTexView<D, T>>> dSubspan = dAttributeSpan.subspan(rangeStart[0],
                                                                        count);
    Span<Optional<TracerTexView<D, T>>> hSpan(hOptTexViews.begin(), hOptTexViews.end());
    assert(hSpan.size() == dSubspan.size());
    queue.MemcpyAsync(dSubspan, ToConstSpan(hSpan));
    // TODO: Try to find a way to remove this wait
    queue.Barrier().Wait();
}

template<class I, class A>
template<uint32_t D, class T>
void GenericTexturedGroupT<I, A>::GenericPushTexAttribute(Span<TracerTexView<D, T>> dAttributeSpan,
                                                          //
                                                          I idStart, I idEnd,
                                                          uint32_t attributeIndex,
                                                          std::vector<TextureId> textureIds,
                                                          const GPUQueue& queue)
{
    auto hTexViews = ConvertToView<D, T>(std::move(textureIds),
                                         attributeIndex);

    // YOLO memcpy here! Hopefully it works
    auto rangeStart = this->FindRange(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->FindRange(idEnd.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<TracerTexView<D, T>> dSubspan = dAttributeSpan.subspan(rangeStart[0],
                                                              count);
    Span<TracerTexView<D, T>> hSpan(hTexViews.cbegin(), hTexViews.cend());
    assert(hSpan.size() == dSubspan.size());
    queue.MemcpyAsync(dSubspan, ToConstSpan(hSpan));
    // TODO: Try to find a way to remove this wait
    queue.Barrier().Wait();
}