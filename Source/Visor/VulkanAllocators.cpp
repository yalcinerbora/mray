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

VulkanDeviceAllocator::VulkanDeviceAllocator(VkDevice d, uint32_t dI, uint32_t hI)
    : deviceVk(d)
    , deviceMemIndex(dI)
    , hostVisibleMemIndex(hI)
{}

VulkanDeviceAllocator& VulkanDeviceAllocator::Instance(VkDevice deviceVk,
                                                       uint32_t deviceMemIndex,
                                                       uint32_t hostVisibleMemIndex)
{
    static VulkanDeviceAllocator allocator;
    if(deviceVk != nullptr)
    {
        allocator = VulkanDeviceAllocator(deviceVk, deviceMemIndex,
                                          hostVisibleMemIndex);
    }
    return allocator;
}
