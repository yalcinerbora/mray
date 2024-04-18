#pragma once

#include <vulkan/vulkan.h>
#include <limits>

#include "Core/Vector.h"
#include "Core/DataStructures.h"

#include "VulkanAllocators.h"

struct VulkanSystemView
{
    VkInstance          instanceVk      = nullptr;
    VkPhysicalDevice    pDeviceVk       = nullptr;
    VkDevice            deviceVk        = nullptr;
    uint32_t            queueIndex      = std::numeric_limits<uint32_t>::max();
    VkQueue             mainQueueVk     = nullptr;
};

class VulkanImage
{
    private:
    VkImage         imgVk       = nullptr;
    VkImageView     viewVk      = nullptr;
    VkSampler       samplerVk   = nullptr;
    const VulkanSystemView* handlesVk = nullptr;

    public:
    // Constructors & Destructor
                    VulkanImage(const VulkanSystemView&);
                    VulkanImage(const VulkanSystemView&,
                                VkFormat format, Vector2i pixRes);
                    VulkanImage(const VulkanImage&) = delete;
                    VulkanImage(VulkanImage&&);
    VulkanImage&    operator=(const VulkanImage&) = delete;
    VulkanImage&    operator=(VulkanImage&&);
                    ~VulkanImage();
    //
    SizeAlignPair   MemRequirements() const;
    void            AttachMemory(VkDeviceMemory, VkDeviceSize);

    void            IssueClear(VkCommandBuffer, VkClearColorValue);

    VkImage         Image() const;
    VkImageView     View() const;
    VkSampler       Sampler() const;
};

class VulkanBuffer
{
    private:
    VkBuffer                bufferVk    = nullptr;
    const VulkanSystemView* handlesVk   = nullptr;
    public:
    // Constructors & Destructor
                    VulkanBuffer(const VulkanSystemView&);
                    VulkanBuffer(const VulkanSystemView&,
                                 VkBufferUsageFlags usageFlags);
                    VulkanBuffer(const VulkanBuffer&) = delete;
                    VulkanBuffer(VulkanBuffer&&);
    VulkanBuffer&   operator=(const VulkanBuffer&) = delete;
    VulkanBuffer&   operator=(VulkanBuffer&&);
                    ~VulkanBuffer();
    //
    SizeAlignPair   MemRequirements() const;
    void            AttachMemory(VkDeviceMemory, VkDeviceSize);
    VkBuffer        Buffer() const;
};


inline VkImage VulkanImage::Image() const
{
    return imgVk;
}

inline VkImageView VulkanImage::View() const
{
    return viewVk;
}

inline VkSampler VulkanImage::Sampler() const
{
    return samplerVk;
}

inline VkBuffer VulkanBuffer::Buffer() const
{
    return bufferVk;
}

static_assert(VulkanMemObjectC<VulkanImage>);
static_assert(VulkanMemObjectC<VulkanBuffer>);