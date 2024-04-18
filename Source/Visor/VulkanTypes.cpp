#include "VulkanTypes.h"
#include "Core/Definitions.h"

VulkanImage::VulkanImage(const VulkanSystemView& handles)
    : handlesVk(&handles)
{}

VulkanImage::VulkanImage(const VulkanSystemView& handles,
                         VkFormat format, Vector2i extent)
    : handlesVk(&handles)
{
    VkImageCreateInfo imgCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent =
        {
            .width = static_cast<uint32_t>(extent[0]),
            .height = static_cast<uint32_t>(extent[1]),
            .depth = 1
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &handlesVk->queueIndex,
        .initialLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    vkCreateImage(handlesVk->deviceVk, &imgCInfo,
                  VulkanHostAllocator::Functions(),
                  &imgVk);

    VkImageViewCreateInfo imgViewCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = imgVk,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components =
        {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY
        },
        .subresourceRange =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    vkCreateImageView(handlesVk->deviceVk, &imgViewCInfo,
                      VulkanHostAllocator::Functions(),
                      &viewVk);
}

VulkanImage::VulkanImage(VulkanImage&& other)
    : imgVk(other.imgVk)
    , viewVk(other.viewVk)
    , handlesVk(other.handlesVk)
{
    other.imgVk = nullptr;
    other.viewVk = nullptr;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other)
{
    assert(this != &other);
    vkDestroyImage(handlesVk->deviceVk, imgVk,
                   VulkanHostAllocator::Functions());
    vkDestroyImageView(handlesVk->deviceVk, viewVk,
                       VulkanHostAllocator::Functions());

    handlesVk = other.handlesVk;
    imgVk = std::exchange(other.imgVk, nullptr);
    viewVk = std::exchange(other.viewVk, nullptr);
    return *this;
}

VulkanImage::~VulkanImage()
{
    if(!imgVk) return;

    vkDestroyImage(handlesVk->deviceVk, imgVk,
                   VulkanHostAllocator::Functions());
    vkDestroyImageView(handlesVk->deviceVk, viewVk,
                       VulkanHostAllocator::Functions());
}

SizeAlignPair VulkanImage::MemRequirements() const
{
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(handlesVk->deviceVk, imgVk,
                                 &requirements);
    return SizeAlignPair
    {
        requirements.alignment,
        requirements.size
    };
}

void VulkanImage::AttachMemory(VkDeviceMemory memVk, VkDeviceSize offset)
{
    vkBindImageMemory(handlesVk->deviceVk, imgVk,
                      memVk, offset);
}

void VulkanImage::IssueClear(VkCommandBuffer cmd, VkClearColorValue color)
{
    vkCmdClearColorImage(cmd, imgVk,
                         VK_IMAGE_LAYOUT_GENERAL, &color, 0, nullptr);
}

VulkanBuffer::VulkanBuffer(const VulkanSystemView& handles)
    : handlesVk(&handles)
{}

VulkanBuffer::VulkanBuffer(const VulkanSystemView& handles,
                           VkBufferUsageFlags usageFlags)
    : handlesVk(&handles)
{
    size_t totalSize = 0;
    VkBufferCreateInfo buffCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
        .size = totalSize,
        .usage = usageFlags,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &handlesVk->queueIndex
    };

    vkCreateBuffer(handlesVk->deviceVk, &buffCInfo,
                   VulkanHostAllocator::Functions(),
                   &bufferVk);
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other)
    : bufferVk(other.bufferVk)
    , handlesVk(other.handlesVk)
{
    other.bufferVk = nullptr;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other)
{
    assert(this != &other);
    handlesVk = other.handlesVk;
    bufferVk = std::exchange(other.bufferVk, nullptr);
    return *this;
}

VulkanBuffer::~VulkanBuffer()
{
    if(!bufferVk) return;
    vkDestroyBuffer(handlesVk->deviceVk, bufferVk,
                    VulkanHostAllocator::Functions());
}

SizeAlignPair VulkanBuffer::MemRequirements() const
{
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(handlesVk->deviceVk,
                                  bufferVk, &requirements);
    return SizeAlignPair
    {
        requirements.alignment,
        requirements.size
    };
}

void VulkanBuffer::AttachMemory(VkDeviceMemory memVk, VkDeviceSize offset)
{
    vkBindBufferMemory(handlesVk->deviceVk, bufferVk,
                       memVk, offset);
}