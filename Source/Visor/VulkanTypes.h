#pragma once

#include <vulkan/vulkan.h>
#include <limits>

#include "Core/Vector.h"
#include "Core/DataStructures.h"

#include "VulkanAllocators.h"

namespace VkConversions
{
    Pair<MRayColorSpaceEnum, Float>
    VkToMRayColorSpace(VkColorSpaceKHR);
}


struct VulkanSystemView
{
    VkInstance          instanceVk          = nullptr;
    VkPhysicalDevice    pDeviceVk           = nullptr;
    VkDevice            deviceVk            = nullptr;
    uint32_t            queueIndex          = std::numeric_limits<uint32_t>::max();
    VkQueue             mainQueueVk         = nullptr;
    VkCommandPool       mainCommandPool     = nullptr;
    VkDescriptorPool    mainDescPool        = nullptr;
};

class VulkanImage
{
    private:
    VkImage         imgVk       = nullptr;
    VkImageView     viewVk      = nullptr;
    VkSampler       samplerVk   = nullptr;
    Vector2ui       extent      = Vector2ui::Zero();
    uint32_t        depth       = 0;
    const VulkanSystemView* handlesVk = nullptr;

    public:
    // Constructors & Destructor
                    VulkanImage(const VulkanSystemView&);
                    VulkanImage(const VulkanSystemView&,
                                VkFormat format,
                                Vector2i pixRes, uint32_t depth = 1);
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
    Vector2ui       Extent() const;

    //
    VkBufferImageCopy FullCopyParams() const;
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
                                 VkBufferUsageFlags usageFlags,
                                 size_t size);
                    VulkanBuffer(const VulkanBuffer&) = delete;
                    VulkanBuffer(VulkanBuffer&&);
    VulkanBuffer&   operator=(const VulkanBuffer&) = delete;
    VulkanBuffer&   operator=(VulkanBuffer&&);
                    ~VulkanBuffer();
    //
    SizeAlignPair   MemRequirements() const;
    void            AttachMemory(VkDeviceMemory, VkDeviceSize);
    VkBuffer        Buffer() const;

    //void            CopyData(Span<const Byte> hRange,
    //                         VkDeviceSize offset);

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

inline Vector2ui VulkanImage::Extent() const
{
    return extent;
}

inline VkBuffer VulkanBuffer::Buffer() const
{
    return bufferVk;
}


static_assert(VulkanMemObjectC<VulkanImage>);
static_assert(VulkanMemObjectC<VulkanBuffer>);