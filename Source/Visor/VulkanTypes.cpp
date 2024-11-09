#include "VulkanTypes.h"
#include "Core/Definitions.h"
#include "Core/Error.h"
#include "Core/Error.hpp"

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

// Constructors & Destructor
VulkanFence::VulkanFence(const VulkanSystemView& view,
                         bool isSignalled)
    : handlesVk(&view)
{
    VkFenceCreateInfo fenceCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = (isSignalled)
                    ? VK_FENCE_CREATE_SIGNALED_BIT
                    : VkFenceCreateFlagBits(0)
    };
    vkCreateFence(handlesVk->deviceVk, &fenceCreateInfo,
                  VulkanHostAllocator::Functions(),
                  &handle);
}

VulkanFence::VulkanFence(VulkanFence&& other)
    : handlesVk(other.handlesVk)
    , handle(std::exchange(other.handle, nullptr))
{}

VulkanFence& VulkanFence::operator=(VulkanFence&& other)
{
    assert(this != &other);
    if(handlesVk)
    {
        vkDestroyFence(handlesVk->deviceVk, handle,
                       VulkanHostAllocator::Functions());
    }
    handlesVk = std::exchange(other.handlesVk, nullptr);
    handle = std::exchange(other.handle, nullptr);
    return *this;
}

VulkanFence::~VulkanFence()
{
    if(!handlesVk) return;
    vkDestroyFence(handlesVk->deviceVk, handle,
                   VulkanHostAllocator::Functions());
}

VulkanFence::operator VkFence()
{
    return handle;
}

VulkanCommandBuffer::VulkanCommandBuffer(const VulkanSystemView& handles)
    : handlesVk(&handles)
{
    VkCommandBufferAllocateInfo cBuffAllocInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = handlesVk->mainCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(handlesVk->deviceVk, &cBuffAllocInfo,
                             &commandBuff);
}

VulkanCommandBuffer::VulkanCommandBuffer(VulkanCommandBuffer&& other)
    : handlesVk(std::exchange(other.handlesVk, nullptr))
    , commandBuff(std::exchange(other.commandBuff, nullptr))
{}

VulkanCommandBuffer& VulkanCommandBuffer::operator=(VulkanCommandBuffer&& other)
{
    assert(this != &other);
    if(handlesVk)
        vkFreeCommandBuffers(handlesVk->deviceVk,
                             handlesVk->mainCommandPool, 1,
                             &commandBuff);
    handlesVk = std::exchange(other.handlesVk, nullptr);
    commandBuff = std::exchange(other.commandBuff, nullptr);
    return *this;
}

VulkanCommandBuffer::~VulkanCommandBuffer()
{
    if(!handlesVk) return;

    vkFreeCommandBuffers(handlesVk->deviceVk,
                         handlesVk->mainCommandPool, 1,
                         &commandBuff);
}

VulkanCommandBuffer::operator VkCommandBuffer()
{
    return commandBuff;
}

VulkanImage::VulkanImage(const VulkanSystemView& handles,
                         VulkanSamplerMode samplerMode,
                         VkFormat format, VkImageUsageFlags usage,
                         Vector2ui extentIn, uint32_t depthIn)
    : handlesVk(&handles)
    , formatVk(format)
    , samplerVk((samplerMode == VulkanSamplerMode::NEAREST)
                        ? handles.nnnSampler
                        : handles.llnSampler)
    , extent(extentIn)
    , depth(depthIn)
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
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &handlesVk->queueIndex,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };
    vkCreateImage(handlesVk->deviceVk, &imgCInfo,
                  VulkanHostAllocator::Functions(),
                  &imgVk);

    formatVk = format;
}

VulkanImage::VulkanImage(VulkanImage&& other)
    : handlesVk(std::exchange(other.handlesVk, nullptr))
    , imgVk(std::exchange(other.imgVk, nullptr))
    , formatVk(other.formatVk)
    , viewVk(std::exchange(other.viewVk, nullptr))
    , samplerVk(other.samplerVk)
    , extent(other.extent)
    , depth(other.depth)
{}

VulkanImage& VulkanImage::operator=(VulkanImage&& other)
{
    assert(this != &other);
    if(handlesVk)
    {
        vkDestroyImage(handlesVk->deviceVk, imgVk,
                       VulkanHostAllocator::Functions());
        vkDestroyImageView(handlesVk->deviceVk, viewVk,
                           VulkanHostAllocator::Functions());
    }

    handlesVk = std::exchange(other.handlesVk, nullptr);
    imgVk = std::exchange(other.imgVk, nullptr);
    formatVk = other.formatVk;
    viewVk = std::exchange(other.viewVk, nullptr);
    samplerVk = other.samplerVk;
    extent = other.extent;
    depth = other.depth;

    return *this;
}

VulkanImage::~VulkanImage()
{
    if(!handlesVk) return;

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

void VulkanImage::CreateView()
{
    VkImageViewCreateInfo imgViewCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = imgVk,
        .viewType = (depth == 1)
                        ? VK_IMAGE_VIEW_TYPE_2D
                        : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = formatVk,
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

VulkanBuffer::VulkanBuffer(const VulkanSystemView& handles,
                           VkBufferUsageFlags usageFlags,
                           size_t sizeIn, bool isForeign)
    : handlesVk(&handles)
    , size(sizeIn)
{
    if(isForeign)
    {
        VkExternalBufferProperties extBufferProps = {};
        extBufferProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
        VkPhysicalDeviceExternalBufferInfo extBufferInfo =
        {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            .pNext = nullptr,
            .flags = 0,
            .usage = usageFlags,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
        };
        vkGetPhysicalDeviceExternalBufferProperties(handlesVk->pDeviceVk,
                                                    &extBufferInfo,
                                                    &extBufferProps);
        auto features = extBufferProps.externalMemoryProperties.externalMemoryFeatures;
        if((features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0)
            throw MRayError("[Visor] External memory is not importable to Vulkan!");
    }

    static constexpr VkExternalMemoryBufferCreateInfo EXT_BUFF_INFO =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
    };
    VkBufferCreateInfo buffCInfo =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = (isForeign) ? &EXT_BUFF_INFO : nullptr,
        .flags = 0,
        .size = size,
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
    : handlesVk(std::exchange(other.handlesVk, nullptr))
    , bufferVk(other.bufferVk)
    , size(other.size)
{
    other.bufferVk = nullptr;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other)
{
    assert(this != &other);
    if(handlesVk)
        vkDestroyBuffer(handlesVk->deviceVk, bufferVk,
                        VulkanHostAllocator::Functions());
    handlesVk = std::exchange(other.handlesVk, nullptr);
    bufferVk = std::exchange(other.bufferVk, nullptr);
    size = other.size;
    return *this;
}

VulkanBuffer::~VulkanBuffer()
{
    if(!handlesVk) return;
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