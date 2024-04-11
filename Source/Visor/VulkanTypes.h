#pragma once

#include <vulkan/vulkan.h>
#include <limits>

struct VulkanSystemView
{
    VkInstance          instanceVk  = nullptr;
    VkPhysicalDevice    pDeviceVk   = nullptr;
    VkDevice            deviceVk    = nullptr;
    uint32_t            queueIndex  = std::numeric_limits<uint32_t>::max();
    VkQueue             mainQueueVk = nullptr;
};