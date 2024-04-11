#include "VisorWindow.h"

#include <GLFW/glfw3.h>
#include <Imgui/imgui_impl_glfw.h>
#include <Imgui/imgui_impl_vulkan.h>
#include <cassert>

#include "Core/Error.h"
#include "Core/MemAlloc.h"

#include "VisorI.h"
#include "VulkanAllocators.h"
#include "VulkanCapabilityFinder.h"
#include "FontAtlas.h"
#include "Visor.h"

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
            return sf.colorSpace == cSpace;
        });
        if(loc == surfaceTypeList.cend())
            MRAY_WARNING_LOG("Unable to create HDR surface, falling back to SDR...");
    }
    //
    if(loc == surfaceTypeList.cend())
        loc = std::find_first_of(surfaceTypeList.cbegin(), surfaceTypeList.cend(),
                                FormatListSDR.cbegin(), FormatListSDR.cend(),
                                [](const VkSurfaceFormatKHR& sf, VkColorSpaceKHR cSpace)
    {
        return sf.colorSpace == cSpace;
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


    MRAY_LOG("=================\n"
             "New swapchain\n"
             "ColorSpace  : {}\n"
             "PresentMode : {}\n"
             "Extent      : [{}, {}]\n"
             "Format Enum : {}\n"
             "=================\n",
             VkColorSpaceToString(colorSpace),
             VkPresentModeToString(presentMode),
             extent.width, extent.height,
             static_cast<VkFlags>(format));

    // Images
    uint32_t requestedImgCount = capabilities.minImageCount + 1;
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
        .CheckVkResultFn = nullptr,
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

    // TODO: Add more for textures
    static const StaticVector<VkDescriptorPoolSize, 1> imguiPoolSizes =
    {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
    };
    VkDescriptorPoolCreateInfo descPoolInfo =
    {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
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

FramebufferPack Swapchain::NextFrame(VkSemaphore imgAvailSem)
{
    uint32_t nextImageIndex;
    VkResult result = vkAcquireNextImageKHR(handlesVk.deviceVk, swapChainVk,
                                            std::numeric_limits<uint64_t>::max(),
                                            imgAvailSem, VK_NULL_HANDLE, &nextImageIndex);
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

void Swapchain::PresentFrame(VkSemaphore waitSingal)
{
    VkPresentInfoKHR presentInfo =
    {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &waitSingal,
        .swapchainCount = 1,
        .pSwapchains = &swapChainVk,
        .pImageIndices = &currentImgIndex,
        .pResults = nullptr
    };
    VkResult result = vkQueuePresentKHR(handlesVk.mainQueueVk, &presentInfo);
    if(result == VK_ERROR_OUT_OF_DATE_KHR ||
       result == VK_SUBOPTIMAL_KHR || fboSizeChanged)
    {
        vkDeviceWaitIdle(handlesVk.deviceVk);
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

void VisorWindow::WndPosChanged(int, int)
{
}

void VisorWindow::WndFBChanged(int newX, int newY)
{
    // TODO: Add more
    assert(newX >= 0);
    assert(newY >= 0);
    swapchain.FBOSizeChanged(Vector2ui(newX, newY));
}

void VisorWindow::WndResized(int, int)
{
}

void VisorWindow::WndClosed()
{
}

void VisorWindow::WndRefreshed()
{
}

void VisorWindow::WndFocused(bool)
{
}

void VisorWindow::WndMinimized(bool)
{
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

MRayError VisorWindow::Initialize(VulkanSystemView handles,
                                  const std::string& windowTitle,
                                  const VisorConfig& config)
{
    handlesVk = handles;
    hdrRequested = config.displayHDR;

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

    glfwShowWindow(window);
    return MRayError::OK;
}

VisorWindow::VisorWindow(VisorWindow&& other)
    : swapchain(std::move(other.swapchain))
    , framePool(std::move(other.framePool))
    , surfaceVk(std::exchange(other.surfaceVk, nullptr))
    , window(std::exchange(other.window, nullptr))
    , hdrRequested(other.hdrRequested)
    , handlesVk(other.handlesVk)
{
    glfwSetWindowUserPointer(window, this);
}

VisorWindow& VisorWindow::operator=(VisorWindow&& other)
{
    assert(this != &other);
    framePool = std::move(other.framePool);
    swapchain = std::move(other.swapchain);

    if(window)
    {
        ImGui_ImplGlfw_Shutdown();
        vkDestroySurfaceKHR(handlesVk.instanceVk, surfaceVk,
                            VulkanHostAllocator::Functions());
        glfwDestroyWindow(window);
    }
    window = std::exchange(other.window, nullptr);
    surfaceVk = std::exchange(other.surfaceVk, nullptr);
    // Tehcnically we do not need to set this
    // by definition, vulkan handles should be the same
    handlesVk = other.handlesVk;

    // Move window user pointer as well
    glfwSetWindowUserPointer(window, this);
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

void VisorWindow::PresentFrame()
{
    return framePool.PresentThisFrame(swapchain);
}

ImFont* VisorWindow::CurrentFont()
{
    float x, y;
    glfwGetWindowContentScale(window, &x, &y);
    assert(x == y);
    return FontAtlas::Instance().GetMonitorFont(x);
}

void VisorWindow::Render()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::PushFont(CurrentFont());

    // GUI Setup
    //---------
    ImGui::ShowDemoWindow();
    //---------

    // IssueCommand
    // -->

    // Rendering
    // ---------

    ImGui::PopFont();
    ImGui::Render();

    // Wait availablility of the command buffer
    FramePack frameHandles = NextFrame();

    VkCommandBufferBeginInfo cbBeginInfo =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr
    };
    vkBeginCommandBuffer(frameHandles.commandBuffer, &cbBeginInfo);

    //
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
        .renderPass = frameHandles.renderPass,
        .framebuffer = frameHandles.fbo,
        .renderArea = VkRect2D
        {
            .offset = {0, 0},
            .extent = frameHandles.extent
        },
        .clearValueCount = 1,
        .pClearValues = &clearValue
    };
    vkCmdBeginRenderPass(frameHandles.commandBuffer, &rpBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);


    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                    frameHandles.commandBuffer);

    vkCmdEndRenderPass(frameHandles.commandBuffer);
    vkEndCommandBuffer(frameHandles.commandBuffer);

    PresentFrame();
}