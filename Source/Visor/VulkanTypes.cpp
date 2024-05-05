#include "VulkanTypes.h"
#include "Core/Definitions.h"

Pair<MRayColorSpaceEnum, Float>
VkConversions::VkToMRayColorSpace(VkColorSpaceKHR cSpace)
{
    // TODO: These are wrong ...
    // Gamma EOTF are not defined by a single float on most cases
    // change this later...
    using RType = Pair<MRayColorSpaceEnum, Float>;
    switch(cSpace)
    {
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            return RType{MRayColorSpaceEnum::MR_REC_709, Float(2.2)};
        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
            return RType{MRayColorSpaceEnum::MR_DCI_P3, Float(2.2)};
        case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
            return RType{MRayColorSpaceEnum::MR_DCI_P3, Float(1.0)};
        case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
            return RType{MRayColorSpaceEnum::MR_DCI_P3, Float(2.2)};
        case VK_COLOR_SPACE_BT709_LINEAR_EXT:
            return RType{MRayColorSpaceEnum::MR_REC_709, Float(1.0)};
        case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
            return RType{MRayColorSpaceEnum::MR_REC_709, Float(2.2)};
        case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
            return RType{MRayColorSpaceEnum::MR_REC_2020, Float(1.0)};
        case VK_COLOR_SPACE_HDR10_ST2084_EXT:
            return RType{MRayColorSpaceEnum::MR_REC_2020, Float(2.2)};
        case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
        case VK_COLOR_SPACE_DOLBYVISION_EXT:
        case VK_COLOR_SPACE_HDR10_HLG_EXT:
        case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
        case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
        case VK_COLOR_SPACE_PASS_THROUGH_EXT:
        case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
        case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
        default: throw MRayError("Unknown VkColorSpaceEnum! ({})",
                                 static_cast<uint32_t>(cSpace));
    }
}

VulkanImage::VulkanImage(const VulkanSystemView& handles)
    : handlesVk(&handles)
{}

VulkanImage::VulkanImage(const VulkanSystemView& handles,
                         VkFormat format, Vector2ui extentIn,
                         uint32_t depthIn)
    : extent(extentIn)
    , depth(depthIn)
    , handlesVk(&handles)
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
            .width = extent[0],
            .height = extent[1],
            .depth = 1
        },
        .mipLevels = 1,
        .arrayLayers = depth,
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
        .viewType = (depth == 1)
                        ? VK_IMAGE_VIEW_TYPE_2D
                        : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
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
            .layerCount = depth
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
    vkCmdClearColorImage(cmd, imgVk, VK_IMAGE_LAYOUT_GENERAL,
                         &color, 0, nullptr);
}

VkBufferImageCopy VulkanImage::FullCopyParams() const
{
    return VkBufferImageCopy
    {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = VkImageSubresourceLayers
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = depth
        },
        .imageOffset = {0u, 0u, 0u},
        .imageExtent = {extent[0], extent[1], 1}
    };
}

VulkanBuffer::VulkanBuffer(const VulkanSystemView& handles)
    : handlesVk(&handles)
{}

VulkanBuffer::VulkanBuffer(const VulkanSystemView& handles,
                           VkBufferUsageFlags usageFlags,
                           size_t size)
    : handlesVk(&handles)
{
    size_t totalSize = size;
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