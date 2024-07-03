#pragma once

#include <vulkan/vulkan_core.h>
#include <map>
#include <memory>
#include <string>
#include "Core/Definitions.h"
#include "Core/Types.h"
#include "Core/Expected.h"

#include "VulkanTypes.h"
#include "MainUniformBuffer.h"

struct MRayError;

struct StagingBufferMemView
{
    VkBuffer        bufferHandle;
    VkDeviceSize    offset;
    VkDeviceSize    size;
};

class StagingMemoryRequesterI
{
    public:
    virtual         ~StagingMemoryRequesterI() = default;
    //
    virtual size_t  StagingBufferSize() const = 0;
    virtual void    SetStagingBufferView(const StagingBufferMemView& stagingBufferView) = 0;
};

class GUITonemapperI
{
    public:
    virtual         ~GUITonemapperI() = default;
    virtual void    Render() = 0;
};

class TonemapperI : public UniformMemoryRequesterI
                  , public StagingMemoryRequesterI
{
    public:
    virtual ~TonemapperI() = default;
    //
    virtual MRayError       Initialize(const std::string& execPath) = 0;
    virtual GUITonemapperI* AcquireGUI() = 0;
    virtual void            TonemapImage(VkCommandBuffer cmd,
                                         const VulkanImage& hdrImg,
                                         const VulkanImage& sdrImg) = 0;
    virtual void            BindImages(const VulkanImage& hdrImg,
                                       const VulkanImage& sdrImg) = 0;

    //
    virtual MRayColorSpaceEnum  InputColorspace() const = 0;
    virtual VkColorSpaceKHR     OutputColorspace() const = 0;
};

class TonemapStage : public UniformMemoryRequesterI
{
    using ShaderKey = Pair<MRayColorSpaceEnum, VkColorSpaceKHR>;
    using TonemapperMap = std::map<ShaderKey, std::unique_ptr<TonemapperI>>;

    private:
    UniformBufferMemView    uniformBuffer = {};
    const VulkanImage*      sdrImage = nullptr;
    const VulkanImage*      hdrImage = nullptr;
    VulkanBuffer            stagingBuffer;
    VulkanDeviceMemory      memory;
    //
    TonemapperMap           tonemappers;
    TonemapperI*            currentTonemapper = nullptr;
    const VulkanSystemView* handlesVk;

    public:
    // Constructors & Destructor
                        TonemapStage(const VulkanSystemView&);
                        TonemapStage(const TonemapStage&) = delete;
                        TonemapStage(TonemapStage&&) = default;
    TonemapStage&       operator=(const TonemapStage&) = delete;
    TonemapStage&       operator=(TonemapStage&&) = default;
                        ~TonemapStage() = default;
    //
    MRayError                   Initialize(const std::string& execPath);
    void                        ChangeImage(const VulkanImage* hdrImageIn,
                                            const VulkanImage* sdrImageIn);
    Expected<GUITonemapperI*>   ChangeTonemapper(MRayColorSpaceEnum renderColorSpace,
                                                 VkColorSpaceKHR swapchainColorSpace);
    // Actions
    void                        IssueTonemap(VkCommandBuffer);

    // A common uniform buffer allocation related
    size_t                      UniformBufferSize() const override;
    void                        SetUniformBufferView(const UniformBufferMemView& uniformBufferPtr) override;

    // Device Memory Related
    SizeAlignPair   MemRequirements() const;
    void            AttachMemory(VkDeviceMemory, VkDeviceSize);
};