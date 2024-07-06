#include "VisorWindow.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_vulkan.h>
#include <cassert>
#include <numeric>

#include "Core/Log.h"
#include "Core/Error.h"
#include "Core/Error.hpp"
#include "Core/MemAlloc.h"

#include "VisorI.h"
#include "VulkanAllocators.h"
#include "VulkanCapabilityFinder.h"
#include "FontAtlas.h"
#include "Visor.h"

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_to_string.hpp>

void ImguiCallback(VkResult vkResult)
{
    if(vkResult == VK_SUCCESS) return;
    MRAY_ERROR_LOG("Imgui VK: {}",
                   vk::to_string(vk::Result(vkResult)));
}

//static constexpr std::array<VkColorSpaceKHR, 16> Colorspaces =
//{
//    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
//    VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT,
//    VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
//    VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT,
//    VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT,
//    VK_COLOR_SPACE_BT709_LINEAR_EXT,
//    VK_COLOR_SPACE_BT709_NONLINEAR_EXT,
//    VK_COLOR_SPACE_BT2020_LINEAR_EXT,
//    VK_COLOR_SPACE_HDR10_ST2084_EXT,
//    VK_COLOR_SPACE_DOLBYVISION_EXT,
//    VK_COLOR_SPACE_HDR10_HLG_EXT,
//    VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT,
//    VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT,
//    VK_COLOR_SPACE_PASS_THROUGH_EXT,
//    VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT,
//    VK_COLOR_SPACE_DISPLAY_NATIVE_AMD
//};

// Debug Helpers
// These are available on validation layer repo
// Currently validation dll comes from SDK
// TODO: Change validation fetching
static std::string VkColorSpaceToString(VkColorSpaceKHR colorSpace)
{
    switch(colorSpace)
    {
        using namespace std::string_literals;
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            return "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR"s;
        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
            return "VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT"s;
        case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
            return "VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT"s;
        case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
            return "VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT"s;
        case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
            return "VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT"s;
        case VK_COLOR_SPACE_BT709_LINEAR_EXT:
            return "VK_COLOR_SPACE_BT709_LINEAR_EXT"s;
        case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
            return "VK_COLOR_SPACE_BT709_NONLINEAR_EXT"s;
        case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
            return "VK_COLOR_SPACE_BT2020_LINEAR_EXT"s;
        case VK_COLOR_SPACE_HDR10_ST2084_EXT:
            return "VK_COLOR_SPACE_HDR10_ST2084_EXT"s;
        case VK_COLOR_SPACE_DOLBYVISION_EXT:
            return "VK_COLOR_SPACE_DOLBYVISION_EXT"s;
        case VK_COLOR_SPACE_HDR10_HLG_EXT:
            return "VK_COLOR_SPACE_HDR10_HLG_EXT"s;
        case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
            return "VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT"s;
        case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
            return "VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT"s;
        case VK_COLOR_SPACE_PASS_THROUGH_EXT:
            return "VK_COLOR_SPACE_PASS_THROUGH_EXT"s;
        case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
            return "VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT"s;
        case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
            return "VK_COLOR_SPACE_DISPLAY_NATIVE_AMD"s;
        default:
            throw MRayError("Unable to convert VkColorSpaceKHR to string!");
    }
}

static std::string VkPresentModeToString(VkPresentModeKHR presentMode)
{
    switch(presentMode)
    {
        case VK_PRESENT_MODE_IMMEDIATE_KHR:
            return "VK_PRESENT_MODE_IMMEDIATE_KHR";
        case VK_PRESENT_MODE_MAILBOX_KHR:
            return "VK_PRESENT_MODE_MAILBOX_KHR";
        case VK_PRESENT_MODE_FIFO_KHR:
            return "VK_PRESENT_MODE_FIFO_KHR";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            return "VK_PRESENT_MODE_FIFO_RELAXED_KHR";
        case VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR:
            return "VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR";
        case VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR:
            return "VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR";
        default:
            throw MRayError("Unable to convert VkPresentModeKHR to string!");
    }
}

static bool VkSurfaceFormatIsSRGB(VkFormat format)
{
    // TODO: Check if Vulkan has this kind of function
    return (format == VK_FORMAT_R8_SRGB ||
            format == VK_FORMAT_R8G8_SRGB ||
            format == VK_FORMAT_R8G8B8_SRGB ||
            format == VK_FORMAT_B8G8R8_SRGB ||
            format == VK_FORMAT_R8G8B8A8_SRGB ||
            format == VK_FORMAT_B8G8R8A8_SRGB ||
            format == VK_FORMAT_A8B8G8R8_SRGB_PACK32);
     // Other formats are Block compressed so these probably are not surface formats
}

// TODO: Refine these?
const std::array<VkColorSpaceKHR, 4> Swapchain::FormatListHDR =
{
    VK_COLOR_SPACE_HDR10_ST2084_EXT,
    VK_COLOR_SPACE_DOLBYVISION_EXT,
    VK_COLOR_SPACE_HDR10_HLG_EXT,
    VK_COLOR_SPACE_PASS_THROUGH_EXT
};

// Sorted in ascending order in terms of color space coverage
// Always non-linear
const std::array<VkColorSpaceKHR, 6> Swapchain::FormatListSDR =
{
    VK_COLOR_SPACE_BT2020_LINEAR_EXT,
    VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT,
    VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT,
    VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT,
    VK_COLOR_SPACE_BT709_NONLINEAR_EXT,
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
};

const std::array<VkPresentModeKHR, 3> Swapchain::PresentModes =
{
    VK_PRESENT_MODE_MAILBOX_KHR,
    VK_PRESENT_MODE_FIFO_RELAXED_KHR,
    VK_PRESENT_MODE_FIFO_KHR
};

FrameCounter::FrameCounter(const VulkanSystemView& sys)
    : handlesVk(&sys)
{
    std::fill(queryData.begin(), queryData.end(), 0u);

    VkQueryPoolCreateInfo createInfo =
    {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
        .pipelineStatistics = 0
    };
    vkCreateQueryPool(handlesVk->deviceVk, &createInfo,
                      VulkanHostAllocator::Functions(), &queryPool);
    vkResetQueryPool(handlesVk->deviceVk, queryPool, 0, 2);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(handlesVk->pDeviceVk, &props);
    timestampPeriod = props.limits.timestampPeriod;

    startCommand = VulkanCommandBuffer(*handlesVk);
    // Since we do not know what will happen in every frame
    // (we have pre-recorded commands for tonemap, accumulate etc.)
    // we also pre-record this and start the frame with this
    VkCommandBufferBeginInfo bI =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        .pInheritanceInfo = nullptr
    };
    vkBeginCommandBuffer(startCommand, &bI);
    vkCmdResetQueryPool(startCommand, queryPool, 0, 2);
    vkCmdWriteTimestamp(startCommand,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        queryPool, 0);
    vkEndCommandBuffer(startCommand);
}

FrameCounter::FrameCounter(FrameCounter&& other)
    : handlesVk(other.handlesVk)
    , queryPool(std::exchange(other.queryPool, nullptr))
    , startCommand(std::move(other.startCommand))
    , queryData(std::exchange(other.queryData, {}))
    , frameCountList(std::exchange(other.frameCountList, {}))
    , firstFrame(other.firstFrame)
    , fillIndex(other.fillIndex)
    , timestampPeriod(other.timestampPeriod)
{}

FrameCounter& FrameCounter::operator=(FrameCounter&& other)
{
    assert(this != &other);

    if(handlesVk)
    {
        vkDestroyQueryPool(handlesVk->deviceVk, queryPool,
                           VulkanHostAllocator::Functions());
    }
    handlesVk = other.handlesVk;
    queryPool = std::exchange(other.queryPool, nullptr);
    startCommand = std::move(other.startCommand);
    queryData = std::exchange(other.queryData, {});
    frameCountList = std::exchange(other.frameCountList, {});
    firstFrame = other.firstFrame;
    fillIndex = other.fillIndex;
    timestampPeriod = other.timestampPeriod;
    return *this;
}

FrameCounter::~FrameCounter()
{
    if(!handlesVk) return;
    vkDestroyQueryPool(handlesVk->deviceVk, queryPool,
                       VulkanHostAllocator::Functions());
}

bool FrameCounter::StartRecord(const VulkanTimelineSemaphore& sem)
{
    // Data is still not filled
    // so do not issue another query
    if(queryData[1] != 0 || queryData[3] != 0) return false;

    auto allStages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo waitSemaphore = sem.WaitInfo(allStages);
    VkSemaphoreSubmitInfo signalSemaphores = sem.SignalInfo(allStages, 1);
    VkCommandBufferSubmitInfo commandSubmitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = startCommand,
        .deviceMask = 0
    };
    VkSubmitInfo2 submitInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitSemaphore,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandSubmitInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalSemaphores
    };
    vkQueueSubmit2(handlesVk->mainQueueVk, 1, &submitInfo, nullptr);
    return true;
}

void FrameCounter::EndRecord(VkCommandBuffer cmd)
{
    // Same as above
    if(queryData[1] != 0 || queryData[3] != 0) return;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        queryPool, 1);
}

float FrameCounter::AvgFrame()
{
    vkGetQueryPoolResults(handlesVk->deviceVk, queryPool,
                          0, 2, 4 * sizeof(uint64_t),
                          queryData.data(), 2 * sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT |
                          VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if(queryData[1] != 0 && queryData[3] != 0)
    {
        double frameTimeMs = static_cast<double>(queryData[2] - queryData[0]);
        frameTimeMs *= (static_cast<double>(timestampPeriod) / 1000000.0);
        float frameTimeMsF = static_cast<float>(frameTimeMs);

        if(firstFrame)
        {
            firstFrame = false;
            std::fill(frameCountList.begin(), frameCountList.end(),
                      static_cast<float>(frameTimeMs));
        }
        else
        {
            frameCountList[fillIndex] = frameTimeMsF;
            fillIndex = MathFunctions::Roll<int32_t>(fillIndex + 1, 0, AVG_FRAME_COUNT);
        }
        // Reset the query availablility
        queryData[1] = queryData[3] = 0;
        vkResetQueryPool(handlesVk->deviceVk, queryPool, 0, 2);
    }
    float result = std::reduce(frameCountList.cbegin(),
                               frameCountList.cend(),
                               0.0f);
    result *= FRAME_COUNT_RECIP;
    return result;
}

MRayError Swapchain::FixSwapchain(bool isFirstFix)
{
    // Wait all commands to complete before resize
    vkDeviceWaitIdle(handlesVk.deviceVk);

    VkSwapchainKHR oldChain = Cleanup(false, !isFirstFix);

    // Capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(handlesVk.pDeviceVk,
                                              surface,
                                              &capabilities);

    // Formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(handlesVk.pDeviceVk, surface,
                                         &formatCount, nullptr);

    surfaceTypeList.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(handlesVk.pDeviceVk, surface,
                                         &formatCount, surfaceTypeList.data());

    // Present Modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(handlesVk.pDeviceVk, surface,
                                            &presentModeCount, nullptr);
    presentModeTypeList.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(handlesVk.pDeviceVk, surface,
                                            &presentModeCount,
                                            presentModeTypeList.data());

    // Surface format mode
    auto loc = surfaceTypeList.cend();
    if(tryHDR)
    {
        loc = std::find_first_of(surfaceTypeList.cbegin(), surfaceTypeList.cend(),
                                FormatListHDR.cbegin(), FormatListHDR.cend(),
                                [](const VkSurfaceFormatKHR& sf, VkColorSpaceKHR cSpace)
        {
            return sf.colorSpace == cSpace && !VkSurfaceFormatIsSRGB(sf.format);
        });
        if(loc == surfaceTypeList.cend() && isFirstFix)
            MRAY_WARNING_LOG("[Visor]: Unable to create HDR surface, falling back to SDR...");
    }
    //
    if(loc == surfaceTypeList.cend())
        loc = std::find_first_of(surfaceTypeList.cbegin(), surfaceTypeList.cend(),
                                FormatListSDR.cbegin(), FormatListSDR.cend(),
                                [](const VkSurfaceFormatKHR& sf, VkColorSpaceKHR cSpace)
    {
        return sf.colorSpace == cSpace && !VkSurfaceFormatIsSRGB(sf.format);
    });
    if(loc == surfaceTypeList.cend())
        return MRayError("Unable to find proper surface format!");
    format = loc->format;
    colorSpace = loc->colorSpace;

    // Present Mode
    auto pMode = std::find_first_of(presentModeTypeList.cbegin(),
                                    presentModeTypeList.cend(),
                                    PresentModes.cbegin(),
                                    PresentModes.cend());
    if(pMode == presentModeTypeList.cend())
        return MRayError("Unable to find proper present mode!");
    presentMode = *pMode;

    // Extent
    bool useApiSize = (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max());
    extent = (useApiSize) ? capabilities.currentExtent
                          : VkExtent2D{fboSize[0], fboSize[0]};
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                            capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                            capabilities.maxImageExtent.height);


    // MRAY_LOG("=================\n"
    //         "New swapchain\n"
    //         "ColorSpace  : {}\n"
    //         "PresentMode : {}\n"
    //         "Extent      : [{}, {}]\n"
    //         "Format Enum : {}\n"
    //         "=================\n",
    //         VkColorSpaceToString(colorSpace),
    //         VkPresentModeToString(presentMode),
    //         extent.width, extent.height,
    //         static_cast<VkFlags>(format));

    // Images
    uint32_t requestedImgCount = capabilities.minImageCount + 1;
    if(capabilities.maxImageCount != 0)
        requestedImgCount = std::min(requestedImgCount, capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR createInfo =
    {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = requestedImgCount ,
        .imageFormat = format,
        .imageColorSpace = colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 999999999,
        .pQueueFamilyIndices = nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = oldChain
    };

    // Finally create the swap chain
    vkCreateSwapchainKHR(handlesVk.deviceVk, &createInfo,
                         VulkanHostAllocator::Functions(),
                         &swapChainVk);

    // Now its ok to destroy old swap chain
    vkDestroySwapchainKHR(handlesVk.deviceVk, oldChain,
                          VulkanHostAllocator::Functions());

     // Finally image related
    uint32_t imageCount;
    vkGetSwapchainImagesKHR(handlesVk.deviceVk,
                            swapChainVk, &imageCount, nullptr);
    images.resize(imageCount);
    vkGetSwapchainImagesKHR(handlesVk.deviceVk, swapChainVk,
                            &imageCount, images.data());


    VkImageViewCreateInfo viewCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = nullptr, // <--- loop will change this
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components = VkComponentMapping
        {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = VkImageSubresourceRange
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    for(const VkImage& img : images)
    {
        viewCreateInfo.image = img;
        VkImageView newView = nullptr;
        vkCreateImageView(handlesVk.deviceVk, &viewCreateInfo,
                          VulkanHostAllocator::Functions(),
                          &newView);
        imageViews.push_back(newView);
    }

     // Renderpass
    VkAttachmentDescription colorAttachment =
    {
        .flags = 0,
        .format = format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };
    VkAttachmentReference colorAttachmentRef =
    {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL

    };
    VkSubpassDescription subpassDesc =
    {
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr
    };
    VkRenderPassCreateInfo renderPassInfo =
    {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpassDesc,
        .dependencyCount = 0,
        .pDependencies = nullptr
    };

    vkCreateRenderPass(handlesVk.deviceVk, &renderPassInfo,
                       VulkanHostAllocator::Functions(),
                       &renderPass);

    // Framebuffers
    VkFramebufferCreateInfo fbCreateInfo =
    {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = renderPass,
        .attachmentCount = 1,
        .pAttachments = nullptr,
        .width = extent.width,
        .height = extent.height,
        .layers = 1
    };
    framebuffers.resize(images.size());
    for(size_t i = 0; i < imageViews.size(); i++)
    {
        fbCreateInfo.pAttachments = &imageViews[i];
        vkCreateFramebuffer(handlesVk.deviceVk, &fbCreateInfo,
                            VulkanHostAllocator::Functions(),
                            &framebuffers[i]);
    }

    assert(images.size() == imageViews.size());
    assert(imageViews.size() == framebuffers.size());

    // TODO: Just because renderpass change (we change renderpass due to
    // swapchain's surface format change). For example user move the window
    // to a HDR screen etc, we need to shutdown imgui then re-init. This feels
    // wrong but w/e.
    ImGui_ImplVulkan_InitInfo imguiInitInfo
    {
        .Instance = handlesVk.instanceVk,
        .PhysicalDevice = handlesVk.pDeviceVk,
        .Device = handlesVk.deviceVk,
        .QueueFamily = handlesVk.queueIndex,
        .Queue = handlesVk.mainQueueVk,
        .DescriptorPool = imguiDescPool,
        .RenderPass = renderPass,
        .MinImageCount = requestedImgCount,
        .ImageCount = imageCount,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        //
        .PipelineCache = nullptr,
        .Subpass = 0,
        //
        .UseDynamicRendering = false,
        .PipelineRenderingCreateInfo = {},
        //
        .Allocator = VulkanHostAllocator::Functions(),
        .CheckVkResultFn = &ImguiCallback,
        .MinAllocationSize = 1_MiB
        //
    };
    ImGui_ImplVulkan_Init(&imguiInitInfo);

    return MRayError::OK;
}

VkSwapchainKHR Swapchain::Cleanup(bool deleteSwapchain,
                                  bool issueImguiShutdown)
{
    if(issueImguiShutdown)
        ImGui_ImplVulkan_Shutdown();

    vkDestroyRenderPass(handlesVk.deviceVk, renderPass,
                        VulkanHostAllocator::Functions());

    for(VkFramebuffer& fbo : framebuffers)
        vkDestroyFramebuffer(handlesVk.deviceVk, fbo,
                             VulkanHostAllocator::Functions());
    framebuffers.clear();

    for(VkImageView& v : imageViews)
        vkDestroyImageView(handlesVk.deviceVk, v,
                           VulkanHostAllocator::Functions());
    imageViews.clear();

    if(deleteSwapchain)
        vkDestroySwapchainKHR(handlesVk.deviceVk, swapChainVk,
                              VulkanHostAllocator::Functions());
    return (deleteSwapchain) ? nullptr : swapChainVk;

}

MRayError Swapchain::Initialize(VulkanSystemView handles,
                                VkSurfaceKHR surf, bool isHDR)
{
    handlesVk   = handles;
    surface     = surf;
    tryHDR      = isHDR;

    static const StaticVector<VkDescriptorPoolSize, 1> imguiPoolSizes =
    {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
    };
    VkDescriptorPoolCreateInfo descPoolInfo =
    {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 16,
        .poolSizeCount = static_cast<uint32_t>(imguiPoolSizes.size()),
        .pPoolSizes = imguiPoolSizes.data()
    };
    vkCreateDescriptorPool(handlesVk.deviceVk, &descPoolInfo,
                           VulkanHostAllocator::Functions(),
                           &imguiDescPool);

    return FixSwapchain(true);
}

Swapchain::Swapchain(Swapchain&& other)
    : handlesVk(other.handlesVk)
    , surface(other.surface)
    , swapChainVk(std::exchange(other.swapChainVk, nullptr))
    , renderPass(std::exchange(other.renderPass, nullptr))
    , images(std::move(other.images))
    , imageViews(std::move(other.imageViews))
    , framebuffers(std::move(other.framebuffers))
    , tryHDR(other.tryHDR)
    , fboSize(other.fboSize)
    , fboSizeChanged(other.fboSizeChanged)
    , currentImgIndex(other.currentImgIndex)
    , presentMode(other.presentMode)
    , colorSpace(other.colorSpace)
    , format(other.format)
    , extent(other.extent)
    , imguiDescPool(std::exchange(other.imguiDescPool, nullptr))
{}

Swapchain& Swapchain::operator=(Swapchain&& other)
{
    assert(&other != this);
    if(swapChainVk)
    {
        Cleanup(true);
        // Cleanup does not handle this
        vkDestroyDescriptorPool(handlesVk.deviceVk, imguiDescPool,
                                VulkanHostAllocator::Functions());
    }

    // These are non-owning
    handlesVk = other.handlesVk;
    surface = other.surface;
    // These needs to be exchanged ("other" should be on "move-from" state)
    swapChainVk = std::exchange(other.swapChainVk, nullptr);
    renderPass = std::exchange(other.renderPass, nullptr);
    imguiDescPool = std::exchange(other.imguiDescPool, nullptr);

    images = std::move(other.images);
    imageViews = std::move(other.imageViews);
    framebuffers = std::move(other.framebuffers);

    tryHDR = other.tryHDR;
    fboSize = other.fboSize;
    fboSizeChanged = other.fboSizeChanged;
    currentImgIndex = other.currentImgIndex;
    presentMode = other.presentMode;
    colorSpace = other.colorSpace;
    format = other.format;
    extent = other.extent;

    return *this;
}

Swapchain::~Swapchain()
{
    if(swapChainVk)
    {
        vkDestroyDescriptorPool(handlesVk.deviceVk, imguiDescPool,
                                VulkanHostAllocator::Functions());
        Cleanup(true);
    }
}

FramebufferPack Swapchain::NextFrame(const VulkanBinarySemaphore& imgAvailSem)
{
    uint32_t nextImageIndex;
    VkResult result = vkAcquireNextImageKHR(handlesVk.deviceVk, swapChainVk,
                                            std::numeric_limits<uint64_t>::max(),
                                            imgAvailSem.Handle(),
                                            VK_NULL_HANDLE, &nextImageIndex);
    if(result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        FixSwapchain();
    }
    else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw MRayError("Fatal error on swapchain recreation");

    currentImgIndex = nextImageIndex % images.size();
    return FramebufferPack
    {
        .extent = extent,
        .img = images[currentImgIndex],
        .imgView = imageViews[currentImgIndex],
        .fbo = framebuffers[currentImgIndex],
        .renderPass = renderPass
    };
}

void Swapchain::PresentFrame(const VulkanBinarySemaphore& waitSingal)
{
    VkSemaphore waitSemHandle = waitSingal.Handle();
    VkPresentInfoKHR presentInfo =
    {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &waitSemHandle,
        .swapchainCount = 1,
        .pSwapchains = &swapChainVk,
        .pImageIndices = &currentImgIndex,
        .pResults = nullptr
    };
    VkResult result = vkQueuePresentKHR(handlesVk.mainQueueVk, &presentInfo);
    if(result == VK_ERROR_OUT_OF_DATE_KHR ||
       result == VK_SUBOPTIMAL_KHR || fboSizeChanged)
    {
        fboSizeChanged = false;
        FixSwapchain();
    }
    else if(result != VK_SUCCESS)
        throw MRayError("Fatal error on swapchain recreation");
}

void Swapchain::FBOSizeChanged(Vector2ui newSize)
{
    fboSize = newSize;
    fboSizeChanged = true;
}

Pair<MRayColorSpaceEnum, Float> Swapchain::ColorSpace() const
{
    return VkConversions::VkToMRayColorSpace(colorSpace);
}

VkColorSpaceKHR Swapchain::ColorSpaceVk() const
{
    return colorSpace;
}

VulkanSystemView VisorWindow::handlesVk = {};

void VisorWindow::WndPosChanged(int, int)
{
}

void VisorWindow::WndFBChanged(int newX, int newY)
{
    assert(newX >= 0 && newY >= 0);
    // Send FBO size change request
    swapchain.FBOSizeChanged(Vector2ui(newX, newY));

    // But do not present if any size is 0
    stopPresenting = (newX == 0 || newY == 0);
}

void VisorWindow::WndResized(int, int)
{
}

void VisorWindow::WndClosed()
{
}

void VisorWindow::WndRefreshed()
{
    // TODO: This feels wrong, (rendering from a callback)
    // but with "waitEvents" instead of poll, currently main loop
    // stays idle (MT runs this blocks).
    Render();
}

void VisorWindow::WndFocused(bool)
{
}

void VisorWindow::WndMinimized(bool isMinimized)
{
    stopPresenting = isMinimized;

    MRAY_LOG("{} presenting!",
             (stopPresenting) ? "Stop" : "Start");
}

void VisorWindow::KeyboardUsed(int /*key*/, int /*scancode*/,
                               int /*action*/, int /*modifier*/)
{
}

void VisorWindow::MouseMoved(double, double)
{
}

void VisorWindow::MousePressed(int /*button*/, int /*action*/,
                               int /*modifier*/)
{
}

void VisorWindow::MouseScrolled(double, double)
{
}

void VisorWindow::PathDropped(int count, const char** paths)
{
    for(int i = 0; i < count; i++)
    {
        MRAY_LOG("Path: {}", paths[i]);
    }
}

MRayError VisorWindow::Initialize(TransferQueue::VisorView& transferQueueIn,
                                  const VulkanSystemView& handles,
                                  TimelineSemaphore* syncSem,
                                  BS::thread_pool* tp,
                                  const std::string& windowTitle,
                                  const VisorConfig& config,
                                  const std::string& processPath)
{
    assert(syncSem != nullptr);
    assert(tp != nullptr);

    transferQueue = &transferQueueIn;
    handlesVk = handles;
    hdrRequested = config.displayHDR;
    threadPool = tp;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(config.wSize[0], config.wSize[1],
                              windowTitle.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    VisorVulkan::RegisterCallbacks(window);

    // Acquire a surface
    if(glfwCreateWindowSurface(handlesVk.instanceVk, window,
                               VulkanHostAllocator::Functions(),
                               &surfaceVk))
    {
        // Window surface creation failed
        return MRayError("Unable to create VkSurface for a GLFW window!");
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(handlesVk.pDeviceVk,
                                         handlesVk.queueIndex,
                                         surfaceVk, &presentSupport);

    // Before initializing the swapchain,
    // Init the ImguiGLFW portion
    ImGui_ImplGlfw_InitForVulkan(window, true);

    MRayError e = swapchain.Initialize(handlesVk, surfaceVk, hdrRequested);
    if(e) return e;

    e = framePool.Initialize(handlesVk);
    if(e) return e;

    frameCounter = FrameCounter(handlesVk);

    // Init Accumulation stage
    e = accumulateStage.Initialize(handlesVk, syncSem, threadPool, processPath);
    if(e) return e;
    // and the Tonemap stage
    e = tonemapStage.Initialize(handlesVk, processPath);
    if(e) return e;
    // Initialize the main semaphore
    imgWriteSem = VulkanTimelineSemaphore(handlesVk);

    // Allocate uniform buffers
    using ReqUBO = UniformMemoryRequesterI;
    auto uboRequesters = std::array<ReqUBO*, 2>{&tonemapStage, &accumulateStage};
    uniformBuffer.AllocateUniformBuffers(uboRequesters);

    // Initially, send the sync semaphore as a very first action
    MRAY_LOG("[Visor]: Sending sync semaphore...");
    transferQueue->Enqueue(VisorAction
    (
        std::in_place_index<VisorAction::SEND_SYNC_SEMAPHORE>,
        syncSem
    ));
    glfwShowWindow(window);
    return MRayError::OK;
}

void VisorWindow::StartRenderpass(const FramePack& frameHandle)
{
    VkClearValue clearValue = {};
    clearValue.color.float32[0] = 1.0f;
    clearValue.color.float32[1] = 0.0f;
    clearValue.color.float32[2] = 0.0f;
    clearValue.color.float32[3] = 0.0f;
    //
    clearValue.depthStencil.depth = 0.0f;
    clearValue.depthStencil.stencil = 0;
    VkRenderPassBeginInfo rpBeginInfo =
    {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = frameHandle.renderPass,
        .framebuffer = frameHandle.fbo,
        .renderArea = VkRect2D
        {
            .offset = {0, 0},
            .extent = frameHandle.extent
        },
        .clearValueCount = 1,
        .pClearValues = &clearValue
    };
    vkCmdBeginRenderPass(frameHandle.commandBuffer, &rpBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
}

void VisorWindow::StartCommandBuffer(const FramePack& frameHandle)
{
    VkCommandBufferBeginInfo cbBeginInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };
    vkBeginCommandBuffer(frameHandle.commandBuffer, &cbBeginInfo);
}

VisorWindow::VisorWindow(VisorWindow&& other)
    : swapchain(std::move(other.swapchain))
    , framePool(std::move(other.framePool))
    , surfaceVk(std::exchange(other.surfaceVk, nullptr))
    , window(std::exchange(other.window, nullptr))
    , hdrRequested(other.hdrRequested)
    , stopPresenting(other.stopPresenting)
    , threadPool(other.threadPool)
    , gui(std::move(other.gui))
    , visorState(other.visorState)
    , frameCounter(std::move(other.frameCounter))
    , transferQueue(other.transferQueue)
    , accumulateStage(std::move(other.accumulateStage))
    , tonemapStage(std::move(other.tonemapStage))
    , renderImagePool(std::move(other.renderImagePool))
    , uniformBuffer(std::move(other.uniformBuffer))
    , imgWriteSem(std::move(other.imgWriteSem))
    , initialSceneFile(other.initialSceneFile)
    , initialTracerRenderConfigPath(other.initialTracerRenderConfigPath)
{
    glfwSetWindowUserPointer(window, this);
}

VisorWindow& VisorWindow::operator=(VisorWindow&& other)
{
    assert(this != &other);
    swapchain = std::move(other.swapchain);
    if(window)
    {
        ImGui_ImplGlfw_Shutdown();
        vkDestroySurfaceKHR(handlesVk.instanceVk, surfaceVk,
                            VulkanHostAllocator::Functions());
        glfwDestroyWindow(window);
    }

    framePool = std::move(other.framePool);
    surfaceVk = std::exchange(other.surfaceVk, nullptr);
    window = std::exchange(other.window, nullptr);
    hdrRequested = other.hdrRequested;
    stopPresenting = other.stopPresenting;
    threadPool = other.threadPool;
    gui = std::move(other.gui);
    visorState = other.visorState;
    frameCounter = std::move(other.frameCounter);
    transferQueue = other.transferQueue;
    accumulateStage = std::move(other.accumulateStage);
    tonemapStage = std::move(other.tonemapStage);
    renderImagePool = std::move(other.renderImagePool);
    uniformBuffer = std::move(other.uniformBuffer);
    imgWriteSem = std::move(other.imgWriteSem);
    initialSceneFile = other.initialSceneFile;
    initialTracerRenderConfigPath = other.initialTracerRenderConfigPath;

    // Move window user pointer as well
    if(window) glfwSetWindowUserPointer(window, this);
    return *this;
}

VisorWindow::~VisorWindow()
{
    // Do not destroy imgui if "moved from" state
    if(window)
    {
        ImGui_ImplGlfw_Shutdown();
        vkDestroySurfaceKHR(handlesVk.instanceVk, surfaceVk,
                            VulkanHostAllocator::Functions());
        glfwDestroyWindow(window);
    }
}

bool VisorWindow::ShouldClose()
{
    return glfwWindowShouldClose(window);
}

FramePack VisorWindow::NextFrame()
{
    return framePool.AcquireNextFrame(swapchain);
}

void VisorWindow::PresentFrame(const VulkanTimelineSemaphore* extraWaitSemaphore)
{
    return framePool.PresentThisFrame(swapchain, extraWaitSemaphore);
}

ImFont* VisorWindow::CurrentFont()
{
    float x, y;
    glfwGetWindowContentScale(window, &x, &y);
    assert(x == y);
    return FontAtlas::Instance().GetMonitorFont(x);
}

void VisorWindow::HandleGUIChanges(const GUIChanges& changes)
{
    // Check the run state
    if(changes.statusBarState.runState)
    {
        visorState.currentRendererState = changes.statusBarState.runState.value();
        TracerRunState state = changes.statusBarState.runState.value();
        switch(state)
        {
            case TracerRunState::RUNNING:
            {
                transferQueue->Enqueue(VisorAction
                (
                    std::in_place_index<VisorAction::START_STOP_RENDER>,
                    true
                ));
                break;
            }
            case TracerRunState::STOPPED:
            {
                transferQueue->Enqueue(VisorAction
                (
                    std::in_place_index<VisorAction::START_STOP_RENDER>,
                    false
                ));
                break;
            }
            case TracerRunState::PAUSED:
            {
                transferQueue->Enqueue(VisorAction
                (
                    std::in_place_index<VisorAction::PAUSE_RENDER>,
                    true
                ));
                break;
            }
            default:
                MRAY_ERROR_LOG("[Visor]: Unkown run state is determined!");
                break;
        }
    }

    if(changes.statusBarState.cameraIndex)
    {
        int32_t camCount = static_cast<int32_t>(visorState.scene.cameraCount);
        int32_t camOffset = changes.statusBarState.cameraIndex.value();
        int32_t newCamIndex = visorState.currentCameraIndex + camOffset;
        newCamIndex = MathFunctions::Roll(newCamIndex, 0, camCount);
        visorState.currentCameraIndex = newCamIndex;

        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::CHANGE_CAMERA>,
            newCamIndex
        ));
    }

    if(changes.transform)
    {
        visorState.transform = changes.transform.value();
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::CHANGE_CAM_TRANSFORM>,
            visorState.transform
        ));
    }

    if(changes.topBarChanges.rendererIndex)
    {
        int32_t rIndex = changes.topBarChanges.rendererIndex.value();
        visorState.currentRenderIndex = rIndex;
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::CHANGE_RENDERER>,
            visorState.tracer.rendererTypes[rIndex]
        ));
    }

    if(changes.topBarChanges.customLogicIndex0)
    {
        int32_t lIndex = changes.topBarChanges.customLogicIndex0.value();
        visorState.currentRenderLogic0 = lIndex;
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::CHANGE_RENDER_LOGIC0>,
            lIndex
        ));
    }

    if(changes.topBarChanges.customLogicIndex1)
    {
        int32_t lIndex = changes.topBarChanges.customLogicIndex1.value();
        visorState.currentRenderLogic1 = lIndex;
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::CHANGE_RENDER_LOGIC1>,
            lIndex
        ));
    }

    if(changes.hdrSaveTrigger)
    {
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::DEMAND_HDR_SAVE>,
            true
        ));
    }

    if(changes.sdrSaveTrigger)
    {
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::DEMAND_SDR_SAVE>,
            true
        ));
    }

    if(changes.visorIsClosed)
        glfwSetWindowShouldClose(window, 1);
}

void VisorWindow::DoInitialActions()
{
    if(initialSceneFile)
    {
        MRAY_LOG("[Visor]: Sending initial scene...");
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::LOAD_SCENE>,
            initialSceneFile.value()
        ));
        initialSceneFile = std::nullopt;
    }

    // Send Initial Renderer once as well
    // This is sent here to start the rendering as well
    if(initialTracerRenderConfigPath)
    {
        MRAY_LOG("[Visor]: Configuring Tracer via initial render config...");

        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::KICKSTART_RENDER>,
            initialTracerRenderConfigPath.value()
        ));

        // Launch the renderer
        transferQueue->Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::START_STOP_RENDER>,
            true
        ));
        // Set the state internally, this will not trigger another send command
        visorState.currentRendererState = TracerRunState::RUNNING;

        initialTracerRenderConfigPath = std::nullopt;
    }
}

size_t VisorWindow::QueryTotalGPUMemory() const
{
    size_t total = 0;
    total += renderImagePool.UsedGPUMemBytes();
    total += tonemapStage.UsedGPUMemBytes();
    total += accumulateStage.UsedGPUMemBytes();
    // Specifically do not add uniform memory since it is "host"
    return total;
}

bool VisorWindow::Render()
{
    Optional<RenderBufferInfo>      newRenderBuffer;
    Optional<RenderImageSection>    newImageSection;
    Optional<bool>                  newClearSignal;
    Optional<RenderImageSaveInfo>   newSaveInfo;
    RenderImagePool::IsHDRImage     isHDRSave = RenderImagePool::SDR;

    TracerResponse response;
    while(transferQueue->TryDequeue(response))
    {
        using RespType = typename TracerResponse::Type;
        RespType tp = static_cast<RespType>(response.index());

        using enum TracerResponse::Type;

        // Stop consuming commands if image section
        // related things are in the queue
        // these require to be processed.
        //
        // For other things, the latest value is enough
        // (most of these are analytics etc)
        bool stopConsuming = false;
        switch(tp)
        {
            case CAMERA_INIT_TRANSFORM:
            {
                MRAY_LOG("[Visor]: Transform received");
                visorState.transform = std::get<CAMERA_INIT_TRANSFORM>(response);
                break;
            }
            case SCENE_ANALYTICS:
            {
                MRAY_LOG("[Visor]: Scene Info received");
                visorState.scene = std::get<SCENE_ANALYTICS>(response);
                break;
            }
            case TRACER_ANALYTICS:
            {
                MRAY_LOG("[Visor]: Tracer Info received");
                visorState.tracer = std::get<TRACER_ANALYTICS>(response);
                break;
            }
            case RENDERER_ANALYTICS:
            {
                //MRAY_LOG("[Visor]: Render Info received");
                visorState.renderer = std::get<RENDERER_ANALYTICS>(response);
                break;
            }
            case RENDERER_OPTIONS:
            {
                MRAY_LOG("[Visor]: Render Options received and ignored");
                break; // TODO: User may change the render options during runtime
            }
            case RENDER_BUFFER_INFO:
            {
                MRAY_LOG("[Visor]: Render Buffer Info received");
                newRenderBuffer = std::get<RENDER_BUFFER_INFO>(response);
                stopConsuming = true;
                break;
            }
            case CLEAR_IMAGE_SECTION:
            {
                MRAY_LOG("[Visor]: Clear Image received");
                newClearSignal = std::get<CLEAR_IMAGE_SECTION>(response);
                stopConsuming = true;
                break;
            }
            case IMAGE_SECTION:
            {
                //MRAY_LOG("[Visor]: Image section received");
                newImageSection = std::get<IMAGE_SECTION>(response);
                stopConsuming = true;
                break;
            }
            case SAVE_AS_HDR:
            {
                MRAY_LOG("[Visor]: Save HDR received");
                newSaveInfo = std::get<SAVE_AS_HDR>(response);
                isHDRSave = RenderImagePool::HDR;
                stopConsuming = true;
                break;
            }
            case SAVE_AS_SDR:
            {
                MRAY_LOG("[Visor]: Save SDR received");
                newSaveInfo = std::get<SAVE_AS_SDR>(response);
                isHDRSave = RenderImagePool::SDR;
                stopConsuming = true;
                break;
            }
            case MEMORY_USAGE:
            {
                MRAY_LOG("[Visor]: Memory usage received");
                visorState.usedGPUMemoryBytes = std::get<MEMORY_USAGE>(response);
                break;
            }
            default: MRAY_WARNING_LOG("[Visor] Unkown tracer response is ignored!"); break;
        }
        if(stopConsuming) break;
    }
    // Current design dictates single operation over the
    // accumulate/save-hdr/save-sdr/renderbufferInfo/clearImage
    // commands. Assert it here, just to be sure
    if constexpr(MRAY_IS_DEBUG)
    {
        std::array<bool, 4> predicates =
        {
            newRenderBuffer.has_value(),
            newImageSection.has_value(),
            newClearSignal.has_value(),
            newSaveInfo.has_value()
        };
        [[maybe_unused]]
        int i = std::transform_reduce(predicates.cbegin(), predicates.cend(),
                                      0, std::plus{},
        [](bool v)
        {
            return v ? 1 : 0;
        });
        assert(i <= 1);
    }

    // Issue frame timer first
    const VulkanTimelineSemaphore* extraWaitSem = nullptr;
    if(!stopPresenting && frameCounter.StartRecord(imgWriteSem))
    {
        imgWriteSem.ChangeNextWait(1);
        extraWaitSem = &imgWriteSem;
    }

    // Entire image reset + img format change (new alloc maybe)
    if(newRenderBuffer)
    {
        // Flush the device, we will need to reallocate
        vkDeviceWaitIdle(handlesVk.deviceVk);

        const auto& newRB = newRenderBuffer.value();
        RenderImageInitInfo renderImageInitParams =
        {
            newRB.resolution,
            newRB.renderColorSpace,
            swapchain.ColorSpace(),
            newRB.depth,
            (newRB.depth > 1)
        };
        renderImagePool = RenderImagePool(threadPool, handlesVk,
                                          renderImageInitParams);

        // Change the images, reset descriptor sets
        accumulateStage.ChangeImage(&renderImagePool.GetHDRImage(),
                                    &renderImagePool.GetSampleImage());
        accumulateStage.ImportExternalHandles(newRB);

        tonemapStage.ChangeImage(&renderImagePool.GetHDRImage(),
                                 &renderImagePool.GetSDRImage());

        // Change the tonemapper if it is new
        auto tonemapperGUI = tonemapStage.ChangeTonemapper(newRB.renderColorSpace,
                                                           swapchain.ColorSpaceVk());
        // A fatal error for visor, we can't tonemap image
        if(tonemapperGUI) throw tonemapperGUI.error();

        // Send GUI the new display image
        gui.ChangeDisplayImage(renderImagePool.GetSDRImage());
        gui.ChangeTonemapperGUI(tonemapperGUI.value());
    }
    if(newSaveInfo)
    {
        auto& rp = renderImagePool;
        rp.SaveImage(isHDRSave, newSaveInfo.value(),
                     imgWriteSem);
        imgWriteSem.ChangeNextWait(2);
    }
    // Before Command Start check if new image section is received
    // Issue an accumulation and override the main wait semaphore
    if(newImageSection)
    {
        auto& as = accumulateStage;
        bool validSubmission = as.IssueAccumulation(newImageSection.value(),
                                                    imgWriteSem);
        // TODO: Abruptly returning here, any synchronization
        // considerations should be checked?
        if(!validSubmission) return false;

        imgWriteSem.ChangeNextWait(1);
    }
    // Image clear
    if(newClearSignal || newRenderBuffer)
    {
        renderImagePool.IssueClear(imgWriteSem);
        imgWriteSem.ChangeNextWait(1);
    }

    // Before tonemap issue check if TM parameters
    // are changed by the user
    GUIChanges guiChanges = gui.Render(CurrentFont(), visorState);
    HandleGUIChanges(guiChanges);

    // Do tonemap
    if(newClearSignal || newRenderBuffer ||
       newImageSection || guiChanges.topBarChanges.newTMParams)
    {
        auto& tm = tonemapStage;
        tm.IssueTonemap(imgWriteSem);
        imgWriteSem.ChangeNextWait(1);
        extraWaitSem = &imgWriteSem;
    }
    // Initially send sync semaphore and initial render config
    DoInitialActions();
    // ============================== //
    //    GUI RENDER/ Command Start   //
    // ============================== //
    if(stopPresenting) return true;
    // Wait availablility of the command buffer
    FramePack frameHandle = NextFrame();
    StartCommandBuffer(frameHandle);
    // ================== //
    //     GUI RENDER     //
    // ================== //
    StartRenderpass(frameHandle);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                    frameHandle.commandBuffer);
    vkCmdEndRenderPass(frameHandle.commandBuffer);
    frameCounter.EndRecord(frameHandle.commandBuffer);
    visorState.visor.frameTime = frameCounter.AvgFrame();
    visorState.visor.usedGPUMemory = QueryTotalGPUMemory();
    vkEndCommandBuffer(frameHandle.commandBuffer);
    PresentFrame(extraWaitSem);
    // Change the timelines next wait, if we used the sem
    if(extraWaitSem) imgWriteSem.ChangeNextWait(1);
    return true;
}

void VisorWindow::SetKickstartParameters(const Optional<std::string_view>& renderConfigPath,
                                         const Optional<std::string_view>& sceneFile)
{
    initialTracerRenderConfigPath = renderConfigPath;
    initialSceneFile = sceneFile;
}