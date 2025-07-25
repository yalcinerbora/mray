#include "VulkanAllocators.h"
#include "VulkanTypes.h"

#include "Core/System.h"
#include "Core/Log.h"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_to_string.hpp>

void* VKAPI_CALL VulkanHostAllocator::Allocate(void*, size_t size, size_t align,
                                               VkSystemAllocationScope)
{
    return AlignedAlloc(size, align);
}

void* VKAPI_CALL VulkanHostAllocator::Realloc(void*, void* ptr,
                                              size_t size, size_t align,
                                              VkSystemAllocationScope)
{
    if(ptr == nullptr)
        return AlignedAlloc(size, align);
    else
        return AlignedRealloc(ptr, size, align);
}

void VKAPI_CALL VulkanHostAllocator::Free(void*, void* ptr)
{
    AlignedFree(ptr, 0, 0);
}

void VKAPI_CALL VulkanHostAllocator::InternalAllocNotify(void*, size_t,
                                                         VkInternalAllocationType,
                                                         VkSystemAllocationScope)
{}

void VKAPI_CALL VulkanHostAllocator::InternalFreeNotify(void*, size_t,
                                                        VkInternalAllocationType,
                                                        VkSystemAllocationScope)
{}

const VkAllocationCallbacks* VulkanHostAllocator::Functions()
{
    static const VkAllocationCallbacks result =
    {
        .pUserData = nullptr,
        .pfnAllocation = &Allocate,
        .pfnReallocation = &Realloc,
        .pfnFree = &Free,
        .pfnInternalAllocation = &InternalAllocNotify,
        .pfnInternalFree = InternalFreeNotify
    };
    return &result;
}

VulkanDeviceAllocator::VulkanDeviceAllocator(VkDevice d, uint32_t dA,
                                             Span<const VkMemoryType> mL,
                                             VkPhysicalDeviceType deviceType)
    : deviceVk(d)
    , deviceCommonAlignment(dA)
{
    // TODO: Remove this when we are confident about memories
    if constexpr(MRAY_IS_DEBUG)
    {
        for(const auto& memType : mL)
        {
            std::string s = vk::to_string(vk::MemoryPropertyFlags(memType.propertyFlags));
            MRAY_DEBUG_LOG("Mem type: {}, HeapIndex: {}", s, memType.heapIndex);
        }
    }
    // Copy to local
    for(const auto& mt : mL) memoryList.push_back(mt);

    // Determine the device local memory and host mapped memory
    // If device is iGPU, get the combo memory,
    // This should be as fast as normal memory (speculation but, I mean come on)
    if(deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
        auto comboMemoryIt = std::find_if(memoryList.begin(),
                                          memoryList.end(),
                                          [](const auto& memType)
        {
            return ((memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                    (memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
        });
        defaultDeviceMemIndex = static_cast<uint32_t>(std::distance(memoryList.begin(),
                                                                    comboMemoryIt));
        defaultHostVisibleMemIndex = defaultDeviceMemIndex;
    }
    // For dGPU's, select two different memories, because it probably be a host pinned
    // memory (In terms of CUDA) so images will not want to be reside there
    else
    {
        auto deviceLocalIt = std::find_if(memoryList.begin(),
                                          memoryList.end(),
                                          [](const auto& memType)
        {
            return memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        });
        // This is guaranteed
        defaultDeviceMemIndex = static_cast<uint32_t>(std::distance(memoryList.begin(),
                                                                    deviceLocalIt));
        // Host visible mem
        auto hostVisibleIt = std::find_if(memoryList.begin(),
                                          memoryList.end(),
                                          [](const auto& memType)
        {
            return memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        });
        defaultHostVisibleMemIndex = static_cast<uint32_t>(std::distance(memoryList.begin(),
                                                                         hostVisibleIt));
    }
    // Finally all done!
    // TODO: Probably not a robust implementation, check back later
    // TODO: what is heap index? Check it
}

VulkanDeviceAllocator& VulkanDeviceAllocator::Instance(VkDevice deviceVk,
                                                       uint32_t deviceAlignment,
                                                       Span<const VkMemoryType> memList,
                                                       VkPhysicalDeviceType deviceType)
{
    static VulkanDeviceAllocator allocator;
    if(deviceVk != nullptr)
    {
        allocator = VulkanDeviceAllocator(deviceVk, deviceAlignment,
                                          memList, deviceType);
    }
    return allocator;
}

VulkanDeviceMemory
VulkanDeviceAllocator::AllocateForeignObject(VulkanBuffer& buffer,
                                             void* foreignPtr,
                                             size_t totalSize,
                                             uint32_t memTypeBits)
{
    std::string s = vk::to_string(vk::MemoryPropertyFlags(memTypeBits));
    MRAY_DEBUG_LOG("Foreign Mem Flags: {}", s);

    auto loc = std::find_if(memoryList.cbegin(), memoryList.cend(),
                            [memTypeBits](const VkMemoryType& memType)
    {
        // Use the first memory that any one of the bits match
        return (memType.propertyFlags & memTypeBits) != 0;
    });

    uint32_t memIndex = std::numeric_limits<uint32_t>::max();
    if(loc == memoryList.cend())
    {
        // Ony my GTX1080, vulkan driver returns 0x300 as property flag?
        // But there is no memory type that supports it.
        //
        // But if I set it to default host visible mem index it works.
        // So put a warning and try it, if it crashes so be it. Current design
        // mandates importing memory so...
        MRAY_WARNING_LOG("[Visor]: Unable to find memory index for the imported host memory "
                         "VkDevice's memory system.\n"
                         "         Trying default selected host visible index [{}].",
                         defaultHostVisibleMemIndex);
        memIndex = defaultHostVisibleMemIndex;
    }
    else memIndex = static_cast<uint32_t>(std::distance(memoryList.cbegin(), loc));

    VkImportMemoryHostPointerInfoEXT hostImportInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = foreignPtr,
    };
    VkMemoryAllocateInfo memAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &hostImportInfo,
        .allocationSize = totalSize,
        .memoryTypeIndex = memIndex
    };
    VulkanDeviceMemory result(deviceVk, memAllocInfo);
    buffer.AttachMemory(result.Memory(), 0);
    return result;
}

VulkanDeviceMemory::VulkanDeviceMemory(VkDevice d, size_t totalSize,
                                       uint32_t memIndex)
    : deviceVk(d)
    , size(totalSize)
{
    VkMemoryAllocateInfo allocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = totalSize,
        .memoryTypeIndex = memIndex
    };
    vkAllocateMemory(deviceVk, &allocInfo,
                     VulkanHostAllocator::Functions(),
                     &memoryVk);
}

VulkanDeviceMemory::VulkanDeviceMemory(VkDevice d,
                                       const VkMemoryAllocateInfo& allocInfo)
    : deviceVk(d)
    , size(allocInfo.allocationSize)
{
    vkAllocateMemory(deviceVk, &allocInfo,
                     VulkanHostAllocator::Functions(),
                     &memoryVk);
}

VulkanDeviceMemory::VulkanDeviceMemory(VulkanDeviceMemory&& other)
    : deviceVk(std::exchange(other.deviceVk, nullptr))
    , memoryVk(std::exchange(other.memoryVk, nullptr))
    , size(other.size)
{}

VulkanDeviceMemory& VulkanDeviceMemory::operator=(VulkanDeviceMemory&& other)
{
    assert(this != &other);
    if(deviceVk)
        vkFreeMemory(deviceVk, memoryVk,
                     VulkanHostAllocator::Functions());

    deviceVk = std::exchange(other.deviceVk, nullptr);
    memoryVk = std::exchange(other.memoryVk, nullptr);
    size = other.size;
    return *this;
}

VulkanDeviceMemory::~VulkanDeviceMemory()
{
    if(!deviceVk) return;
    vkFreeMemory(deviceVk, memoryVk,
                 VulkanHostAllocator::Functions());
}

VkDeviceMemory VulkanDeviceMemory::Memory() const
{
    return memoryVk;
}

size_t VulkanDeviceMemory::SizeBytes() const
{
    return size;
}