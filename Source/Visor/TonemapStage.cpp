#include "TonemapStage.h"
#include <cassert>
#include <type_traits>
#include <utility>
#include <map>
#include "Visor/VulkanTypes.h"

TonemapStage::TonemapStage(const VulkanSystemView& view)
    : uniformBuffer(view)
    , handlesVk(&view)
{}

TonemapStage::TonemapStage(TonemapStage&& other)
    : uniformBuffer(std::move(other.uniformBuffer))
    , sdrImage(std::move(other.sdrImage))
    , tonemappers(std::move(other.tonemappers))
    , currentTonemapper(other.currentTonemapper)
    , handlesVk(other.handlesVk)
{}

TonemapStage& TonemapStage::operator=(TonemapStage&& other)
{
    assert(this != &other);
    uniformBuffer = std::move(other.uniformBuffer);
    tonemappers = std::move(other.tonemappers);
    currentTonemapper = other.currentTonemapper;
    handlesVk = other.handlesVk;
    return *this;
}

TonemapStage::~TonemapStage()
{}

MRayError TonemapStage::Initialize(const std::string& /*execPath*/)
{

    return MRayError::OK;
}

void TonemapStage::ChangeImage(const VulkanImage* hdrImageIn,
                               const VulkanImage* sdrImageIn)
{
    hdrImage = hdrImageIn;
    sdrImage = sdrImageIn;

    // TODO: Invalidate descriptors
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

void TonemapStage::IssueTonemap(VkCommandBuffer)
{
}
