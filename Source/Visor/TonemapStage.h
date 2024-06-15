#pragma once

#include <vulkan/vulkan_core.h>
#include <map>
#include <memory>
#include <string>
#include "Core/Definitions.h"
#include "Core/Types.h"
#include "Core/Expected.h"
#include "VulkanTypes.h"

struct MRayError;

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
    VulkanBuffer            uniformBuffer;
    const VulkanImage*      sdrImage = nullptr;
    const VulkanImage*      hdrImage = nullptr;
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
    void                        ChangeImage(const VulkanImage* hdrImageIn,
                                            const VulkanImage* sdrImageIn);
    Expected<GUITonemapperI*>   ChangeTonemapper(MRayColorSpaceEnum renderColorSpace,
                                                 VkColorSpaceKHR swapchainColorSpace);
    // Actions
    void                        IssueTonemap(VkCommandBuffer);
};