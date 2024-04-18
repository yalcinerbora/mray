#pragma once

#include <vulkan/vulkan.h>
#include <tuple>
#include <array>

#include "Core/Types.h"

using SizeAlignPair = Pair<VkDeviceSize, VkDeviceSize>;

template <class T>
concept VulkanMemObjectC = requires(const T constT, T t)
{
    { constT.MemRequirements() } -> std::same_as<SizeAlignPair>;
    {
        t.AttachMemory(VkDeviceMemory{}, VkDeviceSize{})
    } -> std::same_as<void>;
};

class VulkanHostAllocator
{
    private:
    static void* VKAPI_CALL Allocate(void*, size_t, size_t,
                                     VkSystemAllocationScope);
    static void* VKAPI_CALL Realloc(void*, void*, size_t, size_t,
                                    VkSystemAllocationScope);
    static void VKAPI_CALL  Free(void*, void*);
    static void VKAPI_CALL  InternalAllocNotify(void*, size_t,
                                                VkInternalAllocationType,
                                                VkSystemAllocationScope);
    static void VKAPI_CALL  InternalFreeNotify(void*, size_t,
                                               VkInternalAllocationType,
                                               VkSystemAllocationScope);
    public:
    static const VkAllocationCallbacks* Functions();
};

class VulkanDeviceAllocator
{
    template <size_t N>
    using SizeAlignmentList = std::array<Pair<VkDeviceSize, VkDeviceSize>, N>;

    template <size_t N>
    using OffsetList = std::array<VkDeviceSize, N>;

    private:
    VkDevice deviceVk   = nullptr;
    uint32_t heapIndex  = 0;

    // Constructors & Destructor
    VulkanDeviceAllocator() = default;
    VulkanDeviceAllocator(VkDevice, uint32_t);

    template<size_t I = 0, class... Tp>
    requires (I == sizeof...(Tp))
    void AcquireSizeAndAlignments(SizeAlignmentList<sizeof...(Tp)>&,
                                  const Tuple<Tp&...>&);
    template<std::size_t I = 0, class... Tp>
    requires (I < sizeof...(Tp))
    void AcquireSizeAndAlignments(SizeAlignmentList<sizeof...(Tp)>& sizeAlignmentList,
                                  const Tuple<Tp&...>& inOutObjects);
    //
    template<size_t I = 0, class... Tp>
    requires (I == sizeof...(Tp))
    void AttachMemory(Tuple<Tp&...>&,
                      VkDeviceMemory,
                      const OffsetList<sizeof...(Tp)>&);
    template<std::size_t I = 0, class... Tp>
    requires (I < sizeof...(Tp))
    void AttachMemory(Tuple<Tp&...>& inOutObjects,
                      VkDeviceMemory mem,
                      const OffsetList<sizeof...(Tp)>& offsets);

    public:
    static VulkanDeviceAllocator& Instance(VkDevice deviceVk = nullptr,
                                           uint32_t heapIndex = 0);
    // The alloaction
    template<VulkanMemObjectC... Args>
    VkDeviceMemory AllocateMultiObject(Tuple<Args&...> inOutObjects);
};

template<size_t I, class... Tp>
requires (I == sizeof...(Tp))
void VulkanDeviceAllocator::AcquireSizeAndAlignments(SizeAlignmentList<sizeof...(Tp)>&,
                                                     const Tuple<Tp&...>&)
{}

template<std::size_t I, class... Tp>
requires (I < sizeof...(Tp))
void VulkanDeviceAllocator::AcquireSizeAndAlignments(SizeAlignmentList<sizeof...(Tp)>& sizeAlignmentList,
                                                     const Tuple<Tp&...>& inOutObjects)
{
    sizeAlignmentList[I] = std::get<I>(inOutObjects).MemRequirements();
}

template<size_t I, class... Tp>
requires (I == sizeof...(Tp))
void VulkanDeviceAllocator::AttachMemory(Tuple<Tp&...>&,
                                         VkDeviceMemory mem,
                                         const OffsetList<sizeof...(Tp)>&)
{}

template<std::size_t I, class... Tp>
requires (I < sizeof...(Tp))
void VulkanDeviceAllocator::AttachMemory(Tuple<Tp&...>& inOutObjects,
                                         VkDeviceMemory mem,
                                         const OffsetList<sizeof...(Tp)>& offsets)
{
    std::get<I>(inOutObjects).AttachMemory(mem, offsets[I]);
}

template<VulkanMemObjectC... Args>
VkDeviceMemory VulkanDeviceAllocator::AllocateMultiObject(Tuple<Args&...> inOutObjects)
{
    static constexpr size_t N = sizeof...(Args);
    SizeAlignmentList<N> sizeAndAlignments;
    AcquireSizeAndAlignments(sizeAndAlignments, inOutObjects);

    OffsetList<N> offsets;
    size_t totalSize = 0;
    for(size_t i = 0; i < N; i++)
    {
        offsets[i] = totalSize;
        totalSize = MathFunctions::DivideUp(totalSize,
                                            sizeAndAlignments[i].second);
        totalSize += sizeAndAlignments[i].first;
    }

    VkMemoryAllocateInfo allocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = nullptr,
        .allocationSize = totalSize,
        .memoryTypeIndex = heapIndex
    };
    VkDeviceMemory result;
    vkAllocateMemory(deviceVk, &allocInfo, VulkanHostAllocator::Functions(), &result);

    // Attach the allocated memory to the objects
    AttachMemory(inOutObjects, result, offsets);
    return result;
}


