#include "TonemapStage.h"

TonemapStage::TonemapStage(const VulkanSystemView& view)
    : sdrImage(view)
    , uniformBuffer(view)
    , handlesVk(&view)
{}

TonemapStage::TonemapStage(TonemapStage&& other)
    : sdrImage(std::move(other.sdrImage))
    , uniformBuffer(std::move(other.uniformBuffer))
    , deviceMemVk(std::exchange(other.deviceMemVk, nullptr))
    , tonemappers(std::move(tonemappers))
    , currentTonemapper(other.currentTonemapper)
    , handlesVk(other.handlesVk)
{}

TonemapStage& TonemapStage::operator=(TonemapStage&& other)
{
    assert(this != &other);
    sdrImage = std::move(other.sdrImage);
    uniformBuffer = std::move(other.uniformBuffer);
    deviceMemVk = std::exchange(other.deviceMemVk, nullptr);
    tonemappers = std::move(tonemappers);
    currentTonemapper = other.currentTonemapper;
    handlesVk = other.handlesVk;
    return *this;
}

TonemapStage::~TonemapStage()
{
    if(!deviceMemVk) return;
    vkFreeMemory(handlesVk->deviceVk, deviceMemVk,
                 VulkanHostAllocator::Functions());
}

MRayError TonemapStage::Initialize(const std::string& execPath)
{

    return MRayError::OK;
}

void TonemapStage::ResizeImage(const Vector2i& imgExtent)
{
    sdrImage = VulkanImage(*handlesVk, VK_FORMAT_R16G16B16_SFLOAT,
                           imgExtent);
}

Expected<GUITonemapperI*>
TonemapStage::ChangeTonemapper(MRayColorSpaceEnum renderColorSpace,
                               VkColorSpaceKHR swapchainColorSpace)
{
    auto loc = tonemappers.find({renderColorSpace, swapchainColorSpace});
    if(loc == tonemappers.end())
        return MRayError("Unable to find appropriate tonemapper for RenderImage/Swapchain!");

    currentTonemapper = loc->second.get();
    return currentTonemapper->AcquireGUI();
}

void TonemapStage::TonemapImage(VkCommandBuffer cmd, const VulkanImage& img)
{
    currentTonemapper->TonemapImage(cmd, img, sdrImage);
}

const VulkanImage& TonemapStage::GetImage()
{
    return sdrImage;
}