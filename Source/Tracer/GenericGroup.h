#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <algorithm>

#include "Core/Types.h"
#include "Core/MemAlloc.h"
#include "Device/GPUSystem.h"

#include "TracerTypes.h"

template <class IdType, class AttribInfo>
class GenericGroupI
{
    protected:
    using IdList            = std::vector<IdType>;
    using AttribInfoList    = std::vector<AttribInfo>;

    public:
    virtual         ~GenericGroupI() = default;
    //
    virtual IdList  Reserve(uint32_t itemCount) = 0;
    virtual void    Commit() = 0;
    virtual bool    IsInCommitState() const = 0;
    virtual void    PushAttribute(Vector2ui idRange,
                                  uint32_t attributeIndex,
                                  std::vector<Byte> data) = 0;
    virtual size_t  GPUMemoryUsage() const = 0;
    virtual AttribInfoList AttributeInfo() const = 0;
};

template<class Child, class IdType, class AttribInfo>
class GenericGroup : public GenericGroupI<IdType, AttribInfo>
{
    protected:
    using InterfaceType = GenericGroupI<IdType, AttribInfo>;
    using BaseType = GenericGroup<Child, IdType, AttribInfo>;
    using typename InterfaceType::IdList;
    using typename InterfaceType::AttribInfoList;

    private:
    uint32_t            idCounter;
    size_t              AllocGranularity = 2_MiB;
    size_t              InitialReservation = 32_MiB;
    protected:
    const GPUSystem&    gpuSystem;
    bool                isCommitted;
    uint32_t            groupId;
    DeviceMemory        memory;

    template <class... Args>
    Tuple<Span<Args>...>        GenericCommit();
    template <class T>
    void                        GenericPushData(Vector2ui idRange,
                                                const Span<T>& copyRegion,
                                                std::vector<Byte> data) const;

    public:
                                GenericGroup(uint32_t groupId, const GPUSystem&);
    IdList                      Reserve(uint32_t itemCount) override;
    virtual bool                IsInCommitState() const override;
    virtual size_t              GPUMemoryUsage() const override;
};

template<class C, class ID, class AI>
template <class... Args>
Tuple<Span<Args>...> GenericGroup<C, ID, AI>::GenericCommit()
{
    if(isCommitted)
    {
        throw MRayError(MRAY_FORMAT("{:s} is in committed state, "
                                    " you cannot re-commit!", C::TypeName()));

    }
    size_t totalItemCount = idCounter;

    constexpr size_t TotalElements = sizeof...(Args);
    std::array<size_t, TotalElements> sizes;
    for(size_t i = 0; i < TotalElements; i++)
    {
        sizes[i] = totalItemCount;
    }
    Tuple<Span<Args>...> result;
    MemAlloc::AllocateMultiData<DeviceMemory, Args...>(result, memory, sizes);
    isCommitted = true;
    return result;
}

template<class C, class ID, class AI>
template <class T>
void GenericGroup<C, ID, AI>::GenericPushData(Vector2ui idRange,
                                              const Span<T>& dData,
                                              std::vector<Byte> data) const
{
    // TODO: parallel issue maybe?
    // TODO: utilize multi device maybe
    const GPUQueue& deviceQueue = gpuSystem.BestDevice().GetQueue(0);

    size_t count = idRange[1] - idRange[0];
    Span<T> dSubBatch = dData.subspan(idRange[0], count);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class C, class ID, class AI>
GenericGroup<C, ID, AI>::GenericGroup(uint32_t groupId, const GPUSystem& s)
    : idCounter(0)
    , gpuSystem(s)
    , isCommitted(false)
    , groupId(groupId)
    , memory(gpuSystem.AllGPUs(), AllocGranularity, InitialReservation)
{}

template<class C, class ID, class AI>
typename GenericGroup<C, ID, AI>::IdList GenericGroup<C, ID, AI>::Reserve(uint32_t itemCount)
{
    if(isCommitted)
    {
        throw MRayError(MRAY_FORMAT("{:s} is in committed state, "
                                    " you change cannot change reservations!",
                                    C::TypeName()));
    }
    IdList list(itemCount);
    uint32_t counter = idCounter;
    std::for_each
    (
        list.begin(), list.end(),
        [&](ID& id)
        {
            id = ID::CombinedKey(groupId, counter);
            counter++;
        }
    );
    assert(itemCount == counter);
    idCounter = counter;
    return list;
}

template<class C, class ID, class AI>
bool GenericGroup<C, ID, AI>::IsInCommitState() const
{
    return isCommitted;
}

template<class C, class ID, class AI>
size_t GenericGroup<C, ID, AI>::GPUMemoryUsage() const
{
    return memory.Size();
}