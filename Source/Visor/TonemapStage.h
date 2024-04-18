#pragma once

#include "VulkanTypes.h"
#include "VulkanPipeline.h"
#include <map>

class GUITonemapperI
{
    public:
    virtual         ~GUITonemapperI() = default;
    virtual void    Render() = 0;
};

class TonemapperI
{
    public:
    virtual ~TonemapperI() = default;
    //
    virtual GUITonemapperI* AcquireGUI() = 0;
    virtual void            TonemapImage(VkCommandBuffer cmd,
                                         const VulkanImage& hdrImg,
                                         const VulkanImage& sdrImg) = 0;
};

class TonemapStage
{
    using ShaderKey = Pair<MRayColorSpaceEnum, VkColorSpaceKHR>;
    using TonemapperMap = std::map<ShaderKey, std::unique_ptr<TonemapperI>>;

    private:
    VulkanImage             sdrImage;
    VulkanBuffer            uniformBuffer;
    VkDeviceMemory          deviceMemVk = nullptr;
    //
    TonemapperMap           tonemappers;
    TonemapperI*            currentTonemapper = nullptr;
    const VulkanSystemView* handlesVk;

    public:
    // Constructors & Destructor
                        TonemapStage(const VulkanSystemView&);
                        TonemapStage(const TonemapStage&) = delete;
                        TonemapStage(TonemapStage&&);
    TonemapStage&       operator=(const TonemapStage&) = delete;
    TonemapStage&       operator=(TonemapStage&&);
                        ~TonemapStage();
    //
    MRayError                   Initialize(const std::string& execPath);
    void                        ResizeImage(const Vector2i& imgExtent);
    Expected<GUITonemapperI*>   ChangeTonemapper(MRayColorSpaceEnum renderColorSpace,
                                                 VkColorSpaceKHR swapchainColorSpace);
    void                        TonemapImage(VkCommandBuffer, const VulkanImage& img);
    const VulkanImage&          GetImage();
};