#include "VulkanAllocators.h"
#include "Core/System.h"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_to_string.hpp>

#ifdef MRAY_LINUX
    #include <malloc.h>
#endif

void* VKAPI_CALL VulkanHostAllocator::Allocate(void*, size_t size, size_t align,
                                               VkSystemAllocationScope)
{
    #ifdef MRAY_WINDOWS
        return _aligned_malloc(size, align);
    #elif defined MRAY_LINUX
        return std::aligned_alloc(align, size);
    #endif
}

void* VKAPI_CALL VulkanHostAllocator::Realloc(void*, void* ptr,
                                              size_t size, size_t align,
                                              VkSystemAllocationScope)
{
    #ifdef MRAY_WINDOWS
        return _aligned_realloc(ptr, size, align);
    #elif defined MRAY_LINUX

        //https://stackoverflow.com/questions/64884745/is-there-a-linux-equivalent-of-aligned-realloc
        auto newPtr = std::aligned_alloc(align, size);
        auto oldSize = malloc_usable_size(ptr);
        std::memcpy(newPtr, ptr, oldSize);
        std::free(ptr);
        return newPtr;

    #endif
}

void VKAPI_CALL VulkanHostAllocator::Free(void*, void* ptr)
{
    #ifdef MRAY_WINDOWS
        _aligned_free(ptr);
    #elif defined MRAY_LINUX
        std::free(ptr);
    #endif
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

VulkanDeviceAllocator::VulkanDeviceAllocator(VkDevice d, uint32_t dI, uint32_t hI,
                                             uint32_t dA)
    : deviceVk(d)
    , deviceMemIndex(dI)
    , hostVisibleMemIndex(hI)
    , deviceCommonAlignment(dA)
{}

VulkanDeviceAllocator& VulkanDeviceAllocator::Instance(VkDevice deviceVk,
                                                       uint32_t deviceMemIndex,
                                                       uint32_t hostVisibleMemIndex,
                                                       uint32_t deviceCommonAlignment)
{
    static VulkanDeviceAllocator allocator;
    if(deviceVk != nullptr)
    {
        allocator = VulkanDeviceAllocator(deviceVk, deviceMemIndex,
                                          hostVisibleMemIndex,
                                          deviceCommonAlignment);
    }
    return allocator;
}

VulkanDeviceMemory::VulkanDeviceMemory(VkDevice d)
    : deviceVk(d)
{}

VulkanDeviceMemory::VulkanDeviceMemory(VkDevice d,
                                       size_t totalSize, uint32_t memIndex)
    : deviceVk(d)
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

VulkanDeviceMemory::VulkanDeviceMemory(VulkanDeviceMemory&& other)
    : deviceVk(other.deviceVk)
    , memoryVk(std::exchange(other.memoryVk, nullptr))
{}

VulkanDeviceMemory& VulkanDeviceMemory::operator=(VulkanDeviceMemory&& other)
{
    assert(this != &other);
    if(memoryVk)
        vkFreeMemory(deviceVk, memoryVk,
                     VulkanHostAllocator::Functions());

    deviceVk = other.deviceVk;
    memoryVk = std::exchange(other.memoryVk, nullptr);
    return *this;
}

VulkanDeviceMemory::~VulkanDeviceMemory()
{
    if(memoryVk)
        vkFreeMemory(deviceVk, memoryVk,
                     VulkanHostAllocator::Functions());
}

VkDeviceMemory VulkanDeviceMemory::Memory() const
{
    return memoryVk;
}