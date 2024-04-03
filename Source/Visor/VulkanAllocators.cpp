#include "VulkanAllocators.h"
#include "Core/System.h"

void* VKAPI_CALL VulkanHostAllocator::Allocate(void*, size_t size, size_t align,
                                               VkSystemAllocationScope)
{
    #ifdef MRAY_WINDOWS
        return _aligned_malloc(size, align);
    #elif defined MRAY_LINUX
        return aligned_alloc(size, align);
    #endif
}

void* VKAPI_CALL VulkanHostAllocator::Realloc(void*, void* ptr,
                                              size_t size, size_t align,
                                              VkSystemAllocationScope)
{
    #ifdef MRAY_WINDOWS
        return _aligned_realloc(ptr, size, align);
    #elif defined MRAY_LINUX
    {
        //https://stackoverflow.com/questions/64884745/is-there-a-linux-equivalent-of-aligned-realloc
        auto newPtr = aligned_alloc(size, align);
        auto oldSize = malloc_usable_size(ptr);
        std::memcpy(newPtr, ptr, oldSize);
        std::free(ptr);
    }
    #endif
}

void VKAPI_CALL VulkanHostAllocator::Free(void*, void* ptr)
{
    #ifdef MRAY_WINDOWS
        _aligned_free(ptr);
    #elif defined MRAY_LINUX
        free(ptr);
    #endif
}

void VKAPI_CALL VulkanHostAllocator::InternalAllocNotify(void*, size_t size,
                                                         VkInternalAllocationType type,
                                                         VkSystemAllocationScope scope)
{
}

void VKAPI_CALL VulkanHostAllocator::InternalFreeNotify(void*, size_t size,
                                                        VkInternalAllocationType type,
                                                        VkSystemAllocationScope scope)
{
}

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
